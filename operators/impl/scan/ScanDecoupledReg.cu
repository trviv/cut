// Native CUDA counterpart of ScanDecoupledReg.shader — keep semantics in lockstep.
//
// Register-resident variant of ScanDecoupled. Same single-pass decoupled
// look-back algorithm, same tile geometry (TILE = BLOCK*IPT), same descriptor
// protocol — the ONLY difference is how a tile gets between global memory and
// each thread's IPT-element slice:
//
//   ScanDecoupled     global --striped--> __shared__ sData[TILE] --> registers
//   ScanDecoupledReg  global --vec4-----> registers                (no staging)
//
// The staging buffer exists purely to transpose a coalesced striped load into
// the blocked (contiguous-per-thread) arrangement the sequential slice scan
// needs. It costs BLOCK*IPT*4 bytes of shared memory, which caps residency at
// one block/SM for the larger IPT variants (46 items/thread = 46 KB). Here each
// thread instead reads its own contiguous slice directly with the widest
// aligned vector access, so __shared__ shrinks to a few scalars and occupancy
// becomes register-limited. The trade is access shape: a warp's vec4 loads now
// span 32 separate IPT*4-byte slices instead of one contiguous run, so each
// load instruction issues more L1 transactions even though every fetched byte
// is used and DRAM traffic is identical. Which side wins is a measurement, not
// a derivation — hence a separate variant rather than a rewrite.
//
// Because shared memory no longer bounds the tile, this template also carries
// IPT values (48, 64) that ScanDecoupled cannot express on a 48 KB device.
//
// Element type is the generator's CUT_SCALAR_DTYPE_INPUT (Float32 / Int32 /
// UInt32); look-back descriptors store the value's bit pattern in a uint slot.
//
// State layout (uint buffer), with T = numTiles:
//   state[0*T + tile]  status flag  (0 = NOT_READY, 1 = AGGREGATE, 2 = INCLUSIVE)
//   state[1*T + tile]  tile aggregate        (value bits; written once, immutable)
//   state[2*T + tile]  tile inclusive prefix (value bits; valid once status=INC)
//   state[3*T]         dynamic tile counter
// Aggregate and inclusive live in separate slots so a reader never confuses one
// for the other. __threadfence() orders the value store before the status store
// (and the status load before the value load), so observing a flag guarantees
// observing its value.
//
// NOTE: decoupled look-back relies on concurrent block forward progress. The
// dynamic tile counter makes claim order match launch order so a spinning tile
// only ever waits on tiles that were claimed earlier; validated on the target
// GPU rather than portable to every scheduler.
#include "ComputeOpsShared.h"

#define BLOCK 256
#ifndef IPT
#define IPT 32  // overridden per-variant by the native manifest (-DIPT=N)
#endif
#define TILE (BLOCK * IPT)      // elements per tile (8192 at IPT=32)
#define NUM_WARPS (BLOCK / 32)  // 8

#define FLAG_AGG 1u
#define FLAG_INC 2u

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif
typedef CUT_SCALAR_DTYPE_INPUT scalar_t;

// Widest vector width usable for a thread's slice. The slice base is
// tileStart + tid*IPT elements and tileStart is a multiple of TILE = 256*IPT,
// so the guaranteed element alignment is gcd(IPT, 4).
#if (IPT % 4) == 0
#define VW 4
#define VEC_ALIGN 16
#elif (IPT % 2) == 0
#define VW 2
#define VEC_ALIGN 8
#else
#define VW 1
#define VEC_ALIGN 4
#endif
#define NVEC (IPT / VW)  // vectors per thread slice

// The slice is held as NVEC of these rather than IPT loose scalars, so a load,
// a store, and the register file all see the same shape and no pack/unpack step
// sits between them. Component access stays an unrolled index, so ptxas still
// promotes the whole thing to registers (verified: 0 spill bytes).
struct __align__(VEC_ALIGN) VecT {
    scalar_t s[VW];
};

// Reinterpret the scalar value <-> its uint bit pattern for the descriptor slots.
// For int/uint the value cast preserves the two's-complement bit pattern and
// round-trips exactly; for float it is an explicit bitcast.
#if defined(CUT_DTYPE_INPUT_IS_INT)
__device__ __forceinline__ uint scalarToBits(scalar_t s) { return (uint)s; }
__device__ __forceinline__ scalar_t bitsToScalar(uint u) { return (int)u; }
#elif defined(CUT_DTYPE_INPUT_IS_UINT)
__device__ __forceinline__ uint scalarToBits(scalar_t s) { return (uint)s; }
__device__ __forceinline__ scalar_t bitsToScalar(uint u) { return (uint)u; }
#else
__device__ __forceinline__ uint scalarToBits(scalar_t s) { return __float_as_uint(s); }
__device__ __forceinline__ scalar_t bitsToScalar(uint u) { return __uint_as_float(u); }
#endif

struct PushConstants {
    uint numElements;
    uint isExclusive;
    uint numTiles;
};

extern "C" __global__ void cut_main(const scalar_t* __restrict__ dataIn,
                                    scalar_t* __restrict__ dataOut,
                                    uint* __restrict__ state,
                                    PushConstants pc) {
    __shared__ scalar_t warpTotals[NUM_WARPS];
    __shared__ uint sTile;
    __shared__ scalar_t sExclusive;

    const uint tid = threadIdx.x;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5;
    const uint numTiles = pc.numTiles;

    // Claim a tile id (dynamic, for forward-progress order).
    if (tid == 0) {
        sTile = atomicAdd(&state[3u * numTiles], 1u);
    }
    __syncthreads();

    const uint tile = sTile;
    const uint tileStart = tile * TILE;
    const uint sliceStart = tileStart + tid * IPT;
    // Only the final tile can be partial; every other tile takes the vector
    // path with no per-element bounds check.
    const bool fullTile = (tileStart + TILE) <= pc.numElements;

    // Each thread owns the contiguous slice [tid*IPT, tid*IPT+IPT); load it
    // straight into registers as NVEC vectors, then scan it in place.
    VecT v[NVEC];

    if (fullTile) {
        const VecT* __restrict__ src =
            reinterpret_cast<const VecT*>(dataIn + sliceStart);
        for (uint i = 0; i < NVEC; i++) {
            v[i] = src[i];
        }
    } else {
        for (uint i = 0; i < NVEC; i++) {
            for (uint j = 0; j < VW; j++) {
                const uint g = sliceStart + i * VW + j;
                v[i].s[j] = (g < pc.numElements) ? dataIn[g] : (scalar_t)0;
            }
        }
    }

    // Sequential scan of the slice: run the scan inside each vector, then carry
    // the vector's total into the next one. Same add sequence as a flat scan
    // over IPT scalars, just expressed on the vector shape.
    scalar_t carry = (scalar_t)0;
#pragma unroll
    for (uint i = 0; i < NVEC; i++) {
        v[i].s[0] += carry;
#pragma unroll
        for (uint j = 1; j < VW; j++) {
            v[i].s[j] += v[i].s[j - 1];
        }
        carry = v[i].s[VW - 1];
    }

    scalar_t threadTotal = carry;

    // Block-wide inclusive scan of the per-thread totals (warp shuffle + shared).
    scalar_t x = threadTotal;
#pragma unroll
    for (uint off = 1; off < 32; off <<= 1) {
        scalar_t n = __shfl_up_sync(0xFFFFFFFFu, x, off);
        if (lane >= off) x += n;
    }
    if (lane == 31u) warpTotals[warp] = x;
    __syncthreads();

    scalar_t warpPrefix = (scalar_t)0;
#pragma unroll
    for (uint w = 0; w < NUM_WARPS; w++) {
        scalar_t s = warpTotals[w];
        if (w < warp) warpPrefix += s;
    }
    scalar_t threadPrefix = (x - threadTotal) + warpPrefix; // exclusive prefix of totals
    scalar_t tileAgg = (scalar_t)0;
#pragma unroll
    for (uint w = 0; w < NUM_WARPS; w++) {
        tileAgg += warpTotals[w];                           // whole-tile sum
    }

    // Decoupled look-back on a single thread; broadcast the result via shared.
    if (tid == 0) {
        scalar_t exclusive = (scalar_t)0;
        if (tile == 0u) {
            atomicExch(&state[2u * numTiles + tile], scalarToBits(tileAgg));
            __threadfence();
            atomicExch(&state[tile], FLAG_INC);
        } else {
            atomicExch(&state[numTiles + tile], scalarToBits(tileAgg));
            __threadfence();
            atomicExch(&state[tile], FLAG_AGG);

            int pred = (int)tile - 1;
            while (pred >= 0) {
                uint s;
                do {
                    s = atomicAdd(&state[(uint)pred], 0u);
                } while (s == 0u);
                __threadfence();
                if (s == FLAG_INC) {
                    exclusive += bitsToScalar(
                        atomicAdd(&state[2u * numTiles + (uint)pred], 0u));
                    break;
                }
                exclusive += bitsToScalar(
                    atomicAdd(&state[numTiles + (uint)pred], 0u));
                pred--;
            }
            atomicExch(&state[2u * numTiles + tile],
                       scalarToBits(exclusive + tileAgg));
            __threadfence();
            atomicExch(&state[tile], FLAG_INC);
        }
        sExclusive = exclusive;
    }
    __syncthreads();
    scalar_t base = sExclusive + threadPrefix;

    // Fold the prefix in place. The exclusive form shifts the slice right by
    // one, so walk downwards: the predecessor of a vector's first component is
    // the previous vector's last, which is still untouched at that point.
#pragma unroll
    for (uint i = NVEC; i-- > 0;) {
#pragma unroll
        for (uint j = VW; j-- > 0;) {
            scalar_t prev;
            if (j > 0u)
                prev = v[i].s[j - 1];
            else if (i > 0u)
                prev = v[i - 1].s[VW - 1];
            else
                prev = (scalar_t)0;
            v[i].s[j] = (pc.isExclusive != 0u) ? (base + prev)
                                              : (base + v[i].s[j]);
        }
    }

    // Store the slice back through the same vector shape.
    if (fullTile) {
        VecT* __restrict__ dst = reinterpret_cast<VecT*>(dataOut + sliceStart);
#pragma unroll
        for (uint i = 0; i < NVEC; i++) {
            dst[i] = v[i];
        }
    } else {
#pragma unroll
        for (uint i = 0; i < NVEC; i++) {
#pragma unroll
            for (uint j = 0; j < VW; j++) {
                const uint g = sliceStart + i * VW + j;
                if (g < pc.numElements) {
                    dataOut[g] = v[i].s[j];
                }
            }
        }
    }
}
