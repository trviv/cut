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
// warp owns a contiguous region of the tile and scans it cooperatively, 32
// elements at a time; each warp then folds the per-warp totals below it into its
// own exclusive prefix, while warp 0 walks predecessor tiles — summing their
// aggregates until it hits an INCLUSIVE descriptor — to obtain this tile's
// exclusive prefix. Both are warp-uniform and are added at the store.
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
    __shared__ union {
        uint tile;
        scalar_t exclusive;
    } sSlot;

    const unsigned short tid = threadIdx.x;
    const unsigned short lane = tid & 31u;
    const unsigned short warp = tid >> 5;
    const uint numTiles = pc.numTiles;

    if (tid == 0) {
        sSlot.tile = cuda::atomic_ref<uint, cuda::thread_scope_device>(
                    state[2u * numTiles])
                    .fetch_add(1u, cuda::memory_order_relaxed);
    }
    __syncthreads();

    const uint tile = sSlot.tile;
    const unsigned short threadBase = warp * 32u * IPT + lane;
    // const bool fullTile = (tile * TILE + TILE) <= pc.numElements;
    scalar_t carry = (scalar_t)0;

#pragma unroll
    for (unsigned short i = 0; i < IPT; i++) {
        unsigned short idx = threadBase + i * 32u;
        const uint g = min(tile * TILE + (uint)idx, pc.numElements - 1);
        const scalar_t orig = __ldcs(&dataIn[g]);
        scalar_t x = orig;
        for (unsigned short off = 1; off < 32; off <<= 1) {
            const auto up = cuda::device::warp_shuffle_up(x, off);
            if (up.pred) x += up.data;
        }
        x += carry;
        // Region-local result. The region's own prefix and the tile's are both
        // warp-uniform, so both are added once, at the store.
        sData[idx] = x - ((pc.isExclusive != 0u) ? orig : 0);
        carry = cuda::device::warp_shuffle_idx(x, 31);
    }

    // carry is now this region's total, uniform across the warp.
    if (lane == 0u) warpTotals[warp] = carry;
    __syncthreads();

    ull* desc = DESC(state);
    scalar_t prevBlockExclusive = (scalar_t)0;
    scalar_t warpPrefix = (scalar_t)0;

    // Warp 0 fetches the scan from previous blocks
    if (warp == 0u) {
        // Only needed for 2nd tile and onwards
        if (tile > 0u) {
            prevBlockExclusive = scanWarpLookBack(desc, tile, lane);
            // Only one lane writes to shared mem
            if (lane == 0u) {
                sSlot.exclusive = prevBlockExclusive;
            }
        }
    } else {
        //  Calculate warp prefix for warp after index 0
        for (unsigned short w = 0; w < warp; w++) {
            warpPrefix += warpTotals[w];
        }

        // In the last warp, for 2nd tile and onwards
        // Store the intermediate scan result
        if (warp == NUM_WARPS - 1 && tile > 0u) {
            if (lane == 0u) {
                const scalar_t agg = warpPrefix + carry;
                storeRelease64(&desc[tile], packDesc(FLAG_AGG, agg));
            }
        }
    }

    // For 2nd tile and onwards, get scan results from previous block
    // which is stored in shared mem
    if (tile > 0u) {
        __syncthreads();
        prevBlockExclusive = sSlot.exclusive;
    }

    if (tid == BLOCK - 1) {
        const scalar_t agg = warpPrefix + carry;
        storeRelease64(&desc[tile], packDesc(FLAG_INC, prevBlockExclusive + agg));
    }

    // Same region-striped walk as the scan pass, so this is conflict-free in
    // shared and coalesced in global. Both bias terms are warp-uniform.
    const scalar_t bias = warpPrefix + prevBlockExclusive;
    if ((tile * TILE + TILE) <= pc.numElements) {
        for (uint i = 0; i < IPT; i++) {
            const uint idx = threadBase + i * 32u;
            __stcs(&dataOut[tile * TILE + idx], sData[idx] + bias);
        }
    } else {
        for (uint i = 0; i < IPT; i++) {
            const uint idx = threadBase + i * 32u;
            const uint g = tile * TILE + idx;
            if (g < pc.numElements) {
                __stcs(&dataOut[g], sData[idx] + bias);
            }
        }
    }
}
