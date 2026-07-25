// Native CUDA counterpart of ScanDecoupled.shader — keep semantics in lockstep.
//
// Single-pass prefix scan with decoupled look-back. One block == one tile of
// TILE = BLOCK*IPT elements; the tile id is claimed from a global atomic counter
// (NOT blockIdx) so the chained look-back has a forward-progress order. Each
// thread owns a contiguous IPT-element slice; the tile is scanned locally (a
// per-slice sequential scan plus a warp-shuffle block scan of the per-thread
// totals), then thread 0 walks predecessor tiles — summing their aggregates
// until it hits an INCLUSIVE descriptor — to obtain this tile's exclusive
// prefix, which is broadcast to every element.
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
#define IPT 46  // overridden per-variant by the native manifest (-DIPT=N)
#endif
#define TILE (BLOCK * IPT)      // elements per tile (11776 at IPT=46)
#define NUM_WARPS (BLOCK / 32)  // 8

#define FLAG_AGG 1u
#define FLAG_INC 2u

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif
typedef CUT_SCALAR_DTYPE_INPUT scalar_t;

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
    __shared__ scalar_t sData[TILE];
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

    uint tile = sTile;
    uint tileStart = tile * TILE;

    // Coalesced striped load into natural tile order. The striped mapping keeps
    // every thread on a distinct shared-memory bank (conflict-free) and the
    // compiler already widens the consecutive global reads; an explicit float4
    // path was measured slower here (packing 4 contiguous values per thread into
    // shared reintroduces bank conflicts).
#pragma unroll
    for (uint r = 0; r < IPT; r++) {
        uint i = r * BLOCK + tid;
        uint g = tileStart + i;
        sData[i] = (g < pc.numElements) ? dataIn[g] : (scalar_t)0;
    }
    __syncthreads();

    // Each thread owns the contiguous slice [tid*IPT, tid*IPT+IPT); scan it.
    scalar_t v[IPT];
#pragma unroll
    for (uint r = 0; r < IPT; r++) {
        v[r] = sData[tid * IPT + r];
    }
#pragma unroll
    for (uint r = 1; r < IPT; r++) {
        v[r] += v[r - 1];
    }
    scalar_t threadTotal = v[IPT - 1];

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

    // Fold the prefix into every element; stage back through shared so the
    // global stores stay coalesced (striped).
#pragma unroll
    for (uint r = 0; r < IPT; r++) {
        scalar_t outv = (pc.isExclusive != 0u)
                            ? (base + ((r == 0u) ? (scalar_t)0 : v[r - 1]))
                            : (base + v[r]);
        sData[tid * IPT + r] = outv;
    }
    __syncthreads();
#pragma unroll
    for (uint r = 0; r < IPT; r++) {
        uint i = r * BLOCK + tid;
        uint g = tileStart + i;
        if (g < pc.numElements) {
            dataOut[g] = sData[i];
        }
    }
}
