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
// State layout. Each tile owns ONE 64-bit descriptor, packed as
//   low 32 bits   status flag (0 = NOT_READY, 1 = AGGREGATE, 2 = INCLUSIVE)
//   high 32 bits  the matching value's bits — the tile aggregate while the flag
//                 says AGGREGATE, the inclusive prefix once it says INCLUSIVE
// so a look-back step gets the flag and its value in a SINGLE acquire load. That
// is what makes the packing worth it: three parallel arrays (or even interleaved
// slots) need a second, dependent load to fetch the value the flag just vouched
// for, and that load sits directly in the look-back's critical path.
//
// Packing also removes the need for two value slots. They existed only so a
// reader could never mistake an aggregate for an inclusive prefix; with flag and
// value read as one atomic unit the flag always describes the value beside it, so
// the AGGREGATE -> INCLUSIVE transition can just overwrite the slot.
//
// Descriptors occupy state[0 .. 2*T) viewed as ull — 2*T uint slots, not 3*T,
// since packing collapsed the three parallel arrays into one. The dynamic tile
// counter therefore lands at state[2*T], the first slot past them. ScanOp.cpp
// still allocates 3*T+1 uints because the Vulkan path shares that allocation and
// keeps the unpacked layout; the extra T slots are simply unused here.
//
// Moving the counter off the descriptors' cache line (slot 0 plus a line of
// padding) was measured and is +0.0% at every size: the two only share a line when
// 12*T < 128, i.e. numTiles <= 10, and at <= 10 blocks the counter is bumped <= 10
// times total, so there is no contention to relieve.
//
// NOTE: decoupled look-back relies on concurrent block forward progress. The
// dynamic tile counter makes claim order match launch order so a spinning tile
// only ever waits on tiles that were claimed earlier; validated on the target
// GPU rather than portable to every scheduler.
#include "ComputeOpsShared.h"
#include <cuda/atomic>
#include <cuda/warp>

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
typedef unsigned long long ull;

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

// The per-tile descriptor: {flag, value bits} as one 64-bit word. The state buffer
// is allocated as uint32, so reinterpret it — state is at least 8-byte aligned and
// slot t sits at byte 8*t, so every descriptor is naturally aligned.
#define DESC(state) ((ull*)(state))
__device__ __forceinline__ ull packDesc(uint flag, scalar_t value) {
    return (ull)flag | ((ull)scalarToBits(value) << 32);
}
__device__ __forceinline__ uint descFlag(ull d) { return (uint)d; }
__device__ __forceinline__ scalar_t descValue(ull d) {
    return bitsToScalar((uint)(d >> 32));
}

// Look-back descriptor traffic. The protocol needs a coherent, ordered view of
// other blocks' descriptors. atomicAdd(p, 0) / atomicExch give that, but as a
// read-modify-write at the L2 atomic unit — far heavier than a load or a store.
// The .cg cache modifier is NOT a valid substitute (measured: tile-boundary
// corruption); scoped acquire/release IS. The descriptor is one 64-bit word, so
// a single acquire load carries both the flag and the value it vouches for —
// there is nothing left for a __threadfence() to order.
//
// cuda::atomic_ref at device scope emits exactly that (ld.acquire.gpu /
// st.release.gpu on sm_70+) and handles the older-arch lowering itself, which is
// why there is no __CUDA_ARCH__ dispatch here: hand-written scoped-PTX helpers
// only compile on sm_70+, and NVRTC targets the device's own compute capability,
// so on an older GPU the kernel would fail to build — which this backend
// surfaces as a silently skipped dispatch, not an error.
using DescRef = cuda::atomic_ref<ull, cuda::thread_scope_device>;

__device__ __forceinline__ ull loadAcquire64(ull *p) {
    return DescRef(*p).load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void storeRelease64(ull *p, ull v) {
    DescRef(*p).store(v, cuda::memory_order_release);
}

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
    // The claimed tile id and the look-back's exclusive prefix share one slot:
    // their lifetimes are disjoint and a __syncthreads() separates them. Every
    // thread copies .tile into a register immediately after the claim barrier,
    // well before the look-back writes .exclusive. Reordering either use across
    // the barriers below would make them alias for real.
    __shared__ union {
        uint tile;
        scalar_t exclusive;
    } sSlot;

    const unsigned short tid = threadIdx.x;
    const unsigned short lane = tid & 31u;
    const unsigned short warp = tid >> 5;
    const uint numTiles = pc.numTiles;

    // Claim a tile id (dynamic, for forward-progress order). The counter lives at
    // 2*T, the first slot past the packed descriptors. Relaxed is enough: only the
    // counter's atomicity matters, and the __syncthreads() below is what publishes
    // the result to the rest of the block.
    if (tid == 0) {
        sSlot.tile = cuda::atomic_ref<uint, cuda::thread_scope_device>(
                    state[2u * numTiles])
                    .fetch_add(1u, cuda::memory_order_relaxed);
    }
    __syncthreads();

    const uint tile = sSlot.tile;
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

    v[0] = sData[tid * IPT];
    for (unsigned short r = 1; r < IPT; r++) {
        v[r] = sData[tid * IPT + r];
        v[r] += v[r - 1];
    }

    const scalar_t threadTotal = v[IPT - 1];

    // Block-wide inclusive scan of the per-thread totals (warp shuffle + shared).
    scalar_t x = threadTotal;

    // cuda::device::warp_shuffle_up returns the shuffled value AND the predicate
    // the hardware already set for it: shfl.up's second destination, true exactly
    // when the source lane is in range, which at full warp width IS `lane >= off`.
    // __shfl_up_sync has nowhere to return that, so it forces a redundant compare
    // and a select between the shuffle and the add on every step.
    for (unsigned short off = 1; off < 32; off <<= 1) {
        const auto up = cuda::device::warp_shuffle_up(x, off);
        if (up.pred) x += up.data;
    }
    if (lane == 31u) warpTotals[warp] = x;
    __syncthreads();

    // Scan the NUM_WARPS totals with the same shuffle ladder rather than a serial
    // pass over shared memory. Lane w holds warpTotals[w] and lanes past NUM_WARPS
    // hold 0, so log2(NUM_WARPS) steps leave every lane holding the inclusive
    // prefix of the totals; two broadcasts then hand each thread what it needs.
    // Requires NUM_WARPS <= 32 so the totals fit one lane apiece — true while
    // BLOCK <= 1024. Every warp recomputes this redundantly, which is exactly what
    // avoids a second __syncthreads(): the result lands in registers, not shared.
    scalar_t s = (lane < NUM_WARPS) ? warpTotals[lane] : (scalar_t)0;

    for (unsigned short off = 1; off < NUM_WARPS; off <<= 1) {
        const auto up = cuda::device::warp_shuffle_up(s, off);
        if (up.pred) s += up.data;
    }

    // s at lane w == warpTotals[0] + ... + warpTotals[w]. The exclusive prefix of
    // this thread's warp is therefore lane warp-1's value (0 for warp 0), and the
    // whole-tile sum is the last populated lane. Both shuffles run unconditionally
    // — a warp-uniform branch around one would be legal, but the source lane is
    // cheaper to clamp than the control flow is to reason about. warp_shuffle_idx
    // converts implicitly to the value; its predicate is uninteresting here since
    // both source lanes are always in range.
    const uint prevWarp = (warp == 0u) ? 0u : (uint)warp - 1u;
    const scalar_t prevIncl = cuda::device::warp_shuffle_idx(s, prevWarp);
    const scalar_t warpPrefix = (warp == 0u) ? (scalar_t)0 : prevIncl;
    const scalar_t tileAgg = cuda::device::warp_shuffle_idx(s, NUM_WARPS - 1);
    const scalar_t threadPrefix = (x - threadTotal) + warpPrefix; // exclusive prefix of totals

    if (pc.isExclusive != 0u) {
        sData[tid * IPT] = threadPrefix;
        for (unsigned short r = 1; r < IPT; r++) {
            sData[tid * IPT + r] = threadPrefix + v[r - 1];
        }
    } else {
        for (unsigned short r = 0; r < IPT; r++) {
            sData[tid * IPT + r] = threadPrefix + v[r];
        }
    }

    ull* desc = DESC(state);
    scalar_t prevBlockExclusive = (scalar_t)0;
    scalar_t exclusive = (scalar_t)0;

    if (tile > 0u) {
        // Decoupled look-back on a single thread; broadcast the result via shared.
        if (tid == 0) {
            storeRelease64(&desc[tile], packDesc(FLAG_AGG, tileAgg));

            int pred = (int)tile - 1;
            while (pred >= 0) {
                // One acquire load yields both halves, so the value needs no
                // second dependent load and no fence to pair it with the flag.
                ull d;
                do {
                    d = loadAcquire64(&desc[(uint)pred]);
                } while (descFlag(d) == 0u);
                exclusive += descValue(d);
                if (descFlag(d) == FLAG_INC) break;
                pred--;
            }
            sSlot.exclusive = exclusive;
        }
        __syncthreads();
        prevBlockExclusive = sSlot.exclusive;
    } else {
        __syncthreads();
    }

    if (tid == 0) {
        storeRelease64(&desc[tile], packDesc(FLAG_INC, exclusive + tileAgg));
    }

    for (unsigned short r = 0; r < IPT; r++) {
        unsigned short i = r * BLOCK + tid;
        uint g = tileStart + i;
        if (g < pc.numElements) {
            __stcs(&dataOut[g], sData[i] + prevBlockExclusive);
        }
    }
}
