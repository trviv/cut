// Native CUDA counterpart of ScanDecoupledWT.shader — CUDA-only in practice;
// the .shader exists to give each variant its SPIR-V identity (see ScanOp.h).
//
// ScanDecoupled with a WARP TRANSPOSE between the load and the scan — the
// structure cub::DeviceScan gets from BLOCK_LOAD_WARP_TRANSPOSE. Everything
// after the region scan (warp-totals spine, decoupled look-back, biased store)
// is ScanDecoupled.cu verbatim, so an A/B against ScanIPT<n> isolates the
// exchange and nothing else.
//
// WHY. ScanDecoupled scans in the STRIPED arrangement it loads in, which means
// a full 5-step warp-shuffle ladder PER ITEM — 5 shuffles, 5 adds and a
// broadcast for each of IPT elements, ~154 warp/ALU ops per thread at IPT=14.
// ScanDecoupledReg has the cheap form (one sequential pass over a thread's own
// contiguous slice, IPT-1 adds, then ONE warp scan of the thread totals) but
// pays for it at the memory: a thread's slice is IPT*4 bytes from its
// neighbour's, so every global access touches 32 lines. Nobody had both.
//
// The transpose is what buys both. Load STRIPED (coalesced) into shared, read
// each thread's own contiguous slice back out BLOCKED, scan it sequentially in
// registers, and do one warp scan over the thread totals:
//
//   ScanDecoupled    global --striped--> per-item warp scan --> shared
//   ScanDecoupledReg global --blocked--> sequential scan in registers
//   ScanDecoupledWT  global --striped--> shared --blocked--> sequential scan
//                                  --> shared --striped--> global
//
// ~28 adds + 10 warp ops + 4 shared passes at IPT=14, against ~154 warp/ALU ops
// + 2 shared passes. The trade is two extra passes over shared for ~90 fewer
// warp instructions. Motivation is measured: at N=4M CUT's scan kernel is
// 44.9 us against CUB's 40.8, and it is NOT tile geometry (CUT at tile 2048 is
// still 4 us off CUB at tile 1920), so what is left is per-tile instruction
// cost.
//
// BANK CONFLICTS DECIDE WHICH IPT IS USABLE, and no padding is used. The
// blocked read has lane L reading base + L*IPT + i, so the 32 lanes land on 32
// distinct banks exactly when gcd(IPT, 32) == 1, i.e. when IPT IS ODD; an even
// IPT is a gcd(IPT,32)-way conflict, and a power-of-two IPT is catastrophic
// (IPT=16 is 16-way). This is why CUB's f32 scan policy is 15 items/thread and
// not 16. The striped side is conflict-free either way (consecutive lanes,
// consecutive banks). CUB's alternative — pad one word every 32
// (idx + idx/32) — is cheap on both sides but does NOT fully clear the blocked
// read at odd strides, which is why CUB only enables it for power-of-two
// ITEMS_PER_THREAD. Odd IPT is the better deal here: exactly zero conflicts and
// zero extra shared.
//
// The exchange is WARP-LOCAL — a warp's region is [warp*32*IPT, +32*IPT) and no
// other warp touches it — so both barriers are __syncwarp(), not
// __syncthreads(). Same property ScanDecoupled relies on to read sData back at
// the store with no barrier at all.
//
// State layout, descriptor protocol and the forward-progress note are identical
// to ScanDecoupled.cu, which see.
#include "ComputeOpsShared.h"
#include <cuda/atomic>
#include <cuda/warp>

#define BLOCK 256u
#ifndef IPT
#define IPT 15u  // overridden per-variant by the native manifest (-DIPT=N)
#endif
#define TILE (BLOCK * IPT)
#define NUM_WARPS (BLOCK / 32u)

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
    const unsigned short regionBase = warp * 32u * IPT;
    const unsigned short threadBase = regionBase + lane;

    // 1. STRIPED LOAD into shared, raw. Coalesced in global (consecutive lanes,
    //    consecutive elements) and conflict-free in shared for the same reason.
    //    Out-of-range reads are CLAMPED to the last element rather than zeroed,
    //    exactly as ScanDecoupled does: only the final tile can be partial, it
    //    has no successor to read its aggregate, and every position the garbage
    //    reaches is past numElements and therefore never stored.
#pragma unroll
    for (unsigned short i = 0; i < IPT; i++) {
        const unsigned short idx = threadBase + i * 32u;
        const uint g = min(tile * TILE + idx, pc.numElements - 1u);
        sData[idx] = __ldcs(&dataIn[g]);
    }
    __syncwarp();

    // 2. BLOCKED READ of this thread's own contiguous slice + sequential scan.
    //    IPT-1 adds, no shuffles. v[] must stay in registers, which needs the
    //    unrolled index — without the pragma the array goes to local memory.
    const uint sliceBase = regionBase + (uint)lane * IPT;
    scalar_t v[IPT];
    scalar_t sum = (scalar_t)0;
#pragma unroll
    for (uint i = 0; i < IPT; i++) {
        if (pc.isExclusive != 0u) v[i] = sum;
        sum += sData[sliceBase + i];
        if (pc.isExclusive == 0u) v[i] = sum;
    }

    // 3. ONE warp scan over the per-thread totals — the whole point of the
    //    transpose. cuda::device::warp_shuffle_up returns the predicate the
    //    hardware already set, so no redundant compare per step.
    scalar_t x = sum;
    for (unsigned short off = 1; off < 32; off <<= 1) {
        const auto up = cuda::device::warp_shuffle_up(x, off);
        if (up.pred) x += up.data;
    }
    const scalar_t threadExcl = x - sum;             // exclusive within region
    const scalar_t carry = cuda::device::warp_shuffle_idx(x, 31); // region total

    // 4. BLOCKED WRITE-BACK of the region-local scan. The exclusive form shifts
    //    the slice right by one; v[i-1] is the untouched predecessor, so no
    //    separate copy of the raw values is needed.
#pragma unroll
    for (uint i = 0; i < IPT; i++) {
        sData[sliceBase + i] = threadExcl + v[i];
    }
    __syncwarp();

    // ---- everything below is ScanDecoupled.cu unchanged -------------------
    if (lane == 0u) warpTotals[warp] = carry;
    __syncthreads();

    ull* desc = DESC(state);
    scalar_t prevBlockExclusive = (scalar_t)0;
    scalar_t warpPrefix = (scalar_t)0;

    if (warp == 0u) {
        if (tile > 0u) {
            prevBlockExclusive = scanWarpLookBack(desc, tile, lane);
            if (lane == 0u) {
                sSlot.exclusive = prevBlockExclusive;
            }
        }
    } else {
        for (unsigned short w = 0; w < warp; w++) {
            warpPrefix += warpTotals[w];
        }
        if (warp == NUM_WARPS - 1 && tile > 0u) {
            if (lane == 0u) {
                const scalar_t agg = warpPrefix + carry;
                storeRelease64(&desc[tile], packDesc(FLAG_AGG, agg));
            }
        }
    }

    if (tile > 0u) {
        __syncthreads();
        prevBlockExclusive = sSlot.exclusive;
    }

    if (tid == BLOCK - 1) {
        const scalar_t agg = warpPrefix + carry;
        storeRelease64(&desc[tile], packDesc(FLAG_INC, prevBlockExclusive + agg));
    }

    // Striped walk again: conflict-free in shared, coalesced in global. This is
    // the blocked -> striped half of the transpose (BLOCK_STORE_WARP_TRANSPOSE).
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
