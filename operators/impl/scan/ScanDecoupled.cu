// Native CUDA counterpart of ScanDecoupled.shader — keep semantics in lockstep.
// The two implementations are NOT line-for-line: this one additionally uses
// scoped acquire/release for the look-back, __ldcs/__stcs streaming hints for
// the global load and store, and a warp-parallel look-back. None of the three
// ports — see the note in the .shader for the HLSL equivalents that were tried
// and what they measured.
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

#define BLOCK 256u
#ifndef IPT
#define IPT 46u  // overridden per-variant by the native manifest (-DIPT=N)
#endif
#define TILE (BLOCK * IPT)      // elements per tile (11776 at IPT=46)
#define NUM_WARPS (BLOCK / 32u)  // 8

#include "ScanCommon.cuh"

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

    // Warp-cooperative scan over stride-1 chunks. Warp w owns the contiguous
    // region [w*32*IPT, (w+1)*32*IPT) and walks it 32 elements at a time, so every
    // shared AND global access is lane-consecutive: conflict-free for any IPT, and
    // there is no per-thread v[IPT] register array at all.
    //
    // The global load fuses into this pass. The blocked form staged the tile into
    // shared and read it back for the serial slice scan (2 LDS + 2 STS per
    // element); this is 1 STS here plus the 1 LDS in the store pass.
    //
    // What it costs: log2(32) shuffle steps per element, where the blocked form
    // needed a single add. That is the trade.
    const unsigned short threadBase = warp * 32u * IPT + lane;
    scalar_t carry = (scalar_t)0;
    // Only the last tile can be partial; every other tile then skips the
    // per-element bounds test on both the load and the store.
    const bool fullTile = (tileStart + TILE) <= pc.numElements;

    // #pragma unroll is load-bandwidth-critical, not a micro-opt. The inner
    // shuffle ladder stops ptxas unrolling this loop on its own, and un-unrolled
    // it emits TWO LDG for the whole loop — i.e. at most two global loads in
    // flight per thread, against a 47 KB shared footprint that already pins the
    // block to one per SM. Unrolled, ptxas hoists a rolling window of eight.
#pragma unroll
    for (unsigned short i = 0; i < IPT; i++) {
        unsigned short idx = threadBase + i * 32u;
        const uint g = tileStart + (uint)idx;
        const scalar_t orig =
            (fullTile || g < pc.numElements) ? __ldcs(&dataIn[g]) : (scalar_t)0;

        scalar_t x = orig;
        for (unsigned short off = 1; off < 32; off <<= 1) {
            const auto up = cuda::device::warp_shuffle_up(x, off);
            if (up.pred) x += up.data;
        }
        x += carry;
        // Region-local result. The region's own prefix and the tile's are both
        // warp-uniform, so both are added once, at the store.
        sData[idx] = (pc.isExclusive != 0u) ? (x - orig) : x;
        carry = cuda::device::warp_shuffle_idx(x, 31);
    }

    // carry is now this region's total, uniform across the warp.
    if (lane == 0u) warpTotals[warp] = carry;
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
    // No fold pass: warpPrefix is warp-uniform and is added at the store.

    ull* desc = DESC(state);
    scalar_t prevBlockExclusive = (scalar_t)0;
    scalar_t exclusive = (scalar_t)0;

    if (tile > 0u) {
        // Warp 0 walks the predecessors; ScanCommon.cuh carries the rationale
        // for the windowed, ready-prefix shape. Publishing the AGGREGATE from
        // tid 32 (warp 1) keeps that release store off the walking warp's issue
        // slots.
        if (tid == 32) {
            storeRelease64(&desc[tile], packDesc(FLAG_AGG, tileAgg));
        }
        if (warp == 0u) {
            exclusive = scanWarpLookBack(desc, tile, lane);
            if (lane == 0u)
                sSlot.exclusive = exclusive;
        }
        __syncthreads();
        prevBlockExclusive = sSlot.exclusive;
    }

    if (tid == 0) {
        storeRelease64(&desc[tile], packDesc(FLAG_INC, exclusive + tileAgg));
    }

    // Same region-striped walk as the scan pass, so this is conflict-free in
    // shared and coalesced in global. Both bias terms are warp-uniform.
    const scalar_t bias = warpPrefix + prevBlockExclusive;
    if (fullTile) {
        for (uint i = 0; i < IPT; i++) {
            const uint idx = threadBase + i * 32u;
            __stcs(&dataOut[tileStart + idx], sData[idx] + bias);
        }
    } else {
        for (uint i = 0; i < IPT; i++) {
            const uint idx = threadBase + i * 32u;
            const uint g = tileStart + idx;
            if (g < pc.numElements) {
                __stcs(&dataOut[g], sData[idx] + bias);
            }
        }
    }
}
