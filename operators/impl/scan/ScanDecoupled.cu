// Native CUDA counterpart of ScanDecoupled.shader — keep semantics in lockstep.
// The two implementations are NOT line-for-line: this one additionally uses
// scoped acquire/release for the look-back and __ldcs/__stcs streaming hints for
// the global load and store. Neither ports — see the note in the .shader for the
// HLSL equivalents that were tried and what they measured.
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

// Look-back descriptor traffic. The protocol needs a coherent, ordered view of
// other blocks' descriptors. atomicAdd(p, 0) / atomicExch give that, but as a
// read-modify-write at the L2 atomic unit — far heavier than a load or a store.
// The .cg cache modifier is NOT a valid substitute (measured: tile-boundary
// corruption); scoped acquire/release IS, and it also subsumes the separate
// __threadfence(), since release orders the value store before the flag store
// and acquire orders the flag load before the value load.
// Scoped acquire/release is sm_70+. NVRTC compiles for the device's own compute
// capability, so an older GPU would otherwise fail to compile the kernel — which
// this backend surfaces as a silently skipped dispatch, not an error. Fall back
// to the atomic form there; it is slower but identical in meaning.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define CUT_SCAN_HAS_ACQREL 1
__device__ __forceinline__ uint loadAcquire(const uint *p) {
    uint r;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(r) : "l"(p) : "memory");
    return r;
}
__device__ __forceinline__ uint loadRelaxed(const uint *p) {
    uint r;
    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(r) : "l"(p) : "memory");
    return r;
}
__device__ __forceinline__ void storeRelaxed(uint *p, uint v) {
    asm volatile("st.relaxed.gpu.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void storeRelease(uint *p, uint v) {
    asm volatile("st.release.gpu.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
#else
__device__ __forceinline__ uint loadAcquire(uint *p) {
    uint r = atomicAdd(p, 0u);
    __threadfence();
    return r;
}
__device__ __forceinline__ uint loadRelaxed(uint *p) { return atomicAdd(p, 0u); }
__device__ __forceinline__ void storeRelaxed(uint *p, uint v) {
    atomicExch(p, v);
}
__device__ __forceinline__ void storeRelease(uint *p, uint v) {
    __threadfence();
    atomicExch(p, v);
}
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
    const unsigned short lane = tid & 31u;
    const unsigned short warp = tid >> 5;
    const uint numTiles = pc.numTiles;

    // Claim a tile id (dynamic, for forward-progress order).
    if (tid == 0) {
        sTile = atomicAdd(&state[3u * numTiles], 1u);
    }
    __syncthreads();

    const uint tile = sTile;
    const uint tileStart = tile * TILE;

    // Coalesced striped load into natural tile order. The striped mapping keeps
    // every thread on a distinct shared-memory bank (conflict-free) and the
    // compiler already widens the consecutive global reads; an explicit float4
    // path was measured slower here (packing 4 contiguous values per thread into
    // shared reintroduces bank conflicts).
    for (unsigned short r = 0; r < IPT; r++) {
        const unsigned short i = r * BLOCK + tid;
        const uint g = tileStart + i;
        sData[i] = (g < pc.numElements) ? __ldcs(&dataIn[g]) : (scalar_t)0;
    }
    __syncthreads();

    // Each thread owns the contiguous slice [tid*IPT, tid*IPT+IPT); scan it.
    scalar_t v[IPT];

    for (unsigned short r = 0; r < IPT; r++) {
        v[r] = sData[tid * IPT + r];
    }

    // Sequential scan of the slice, two elements per step: v[r+1] is folded from
    // v[r-1] and the ORIGINAL v[r], so it does not wait on v[r]'s own update —
    // half the dependent-add chain of a plain running sum. The bound is r+1 <
    // IPT, not r < IPT: IPT is even, so r < IPT would let the last iteration
    // write v[IPT] and run off the end of the register array. That leaves the
    // final element unpaired, folded after the loop.
    for (uint r = 1; r + 1 < IPT; r += 2) {
        const scalar_t val = v[r - 1];
        v[r + 1] += val + v[r];
        v[r] += val;
    }
    v[IPT - 1] += v[IPT - 2];

    const scalar_t threadTotal = v[IPT - 1];

    // Block-wide inclusive scan of the per-thread totals (warp shuffle + shared).
    scalar_t x = threadTotal;

    for (uint off = 1; off < 32; off <<= 1) {
        const scalar_t n = __shfl_up_sync(0xFFFFFFFFu, x, off);
        x += lane >= off ? n : 0;
    }
    if (lane == 31u) warpTotals[warp] = x;
    __syncthreads();

    scalar_t warpPrefix = (scalar_t)0;

    for (uint w = 0; w < NUM_WARPS; w++) {
        scalar_t s = warpTotals[w];
        if (w < warp) warpPrefix += s;
    }
    const scalar_t threadPrefix = (x - threadTotal) + warpPrefix; // exclusive prefix of totals
    scalar_t tileAgg = (scalar_t)0;

    for (uint w = 0; w < NUM_WARPS; w++) {
        tileAgg += warpTotals[w];                           // whole-tile sum
    }

    // Decoupled look-back on a single thread; broadcast the result via shared.
    if (tid == 0) {
        scalar_t exclusive = (scalar_t)0;
        if (tile == 0u) {
            storeRelaxed(&state[2u * numTiles + tile], scalarToBits(tileAgg));
            storeRelease(&state[tile], FLAG_INC);
        } else {
            storeRelaxed(&state[numTiles + tile], scalarToBits(tileAgg));
            storeRelease(&state[tile], FLAG_AGG);

            int pred = (int)tile - 1;
            while (pred >= 0) {
                uint s;
                do {
                    s = loadAcquire(&state[(uint)pred]);
                } while (s == 0u);
                if (s == FLAG_INC) {
                    exclusive += bitsToScalar(
                        loadRelaxed(&state[2u * numTiles + (uint)pred]));
                    break;
                }
                exclusive +=
                    bitsToScalar(loadRelaxed(&state[numTiles + (uint)pred]));
                pred--;
            }
            storeRelaxed(&state[2u * numTiles + tile],
                         scalarToBits(exclusive + tileAgg));
            storeRelease(&state[tile], FLAG_INC);
        }
        sExclusive = exclusive;
    }
    __syncthreads();
    const scalar_t base = sExclusive + threadPrefix;

    // Fold the prefix into every element; stage back through shared so the
    // global stores stay coalesced (striped).
    if (pc.isExclusive != 0u) {
        sData[tid * IPT] = base;
        for (unsigned short r = 1; r < IPT; r++) {
            sData[tid * IPT + r] = base + v[r - 1];
        }
    } else {
        for (unsigned short r = 0; r < IPT; r++) {
            sData[tid * IPT + r] = base + v[r];
        }
    }

    __syncthreads();

    for (unsigned short r = 0; r < IPT; r++) {
        unsigned short i = r * BLOCK + tid;
        uint g = tileStart + i;
        if (g < pc.numElements) {
            __stcs(&dataOut[g], sData[i]);
        }
    }
}
