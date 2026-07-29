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
// State layout — identical to ScanDecoupled.cu, which see for the rationale.
// Each tile owns ONE 64-bit descriptor: low 32 bits the status flag (0 =
// NOT_READY, 1 = AGGREGATE, 2 = INCLUSIVE), high 32 bits the matching value's
// bits. A look-back step therefore gets flag and value in a single acquire load,
// with no second dependent load in the critical path and nothing left for a
// __threadfence() to order. Descriptors occupy state[0 .. 2*T) viewed as ull and
// the dynamic tile counter lands at state[2*T], so the whole buffer is 2*T + 1
// uints — which is what ScanOp.cpp allocates.
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
#define IPT 32  // overridden per-variant by the native manifest (-DIPT=N)
#endif
#define TILE (BLOCK * IPT)      // elements per tile (8192 at IPT=32)
#define NUM_WARPS (BLOCK / 32)  // 8

#include "ScanCommon.cuh"

// Widest vector width usable for a thread's slice. The slice base is
// tileStart + tid*IPT elements and tileStart is a multiple of TILE = 256*IPT,
// so the guaranteed element alignment is gcd(IPT, 4). Vector staging is this
// variant's whole point, so it stays here rather than in ScanCommon.cuh.
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

    const unsigned short tid = threadIdx.x;
    const unsigned short lane = tid & 31u;
    const unsigned short warp = tid >> 5;
    const uint numTiles = pc.numTiles;

    // Claim a tile id (dynamic, for forward-progress order). The counter sits at
    // 2*T, the first slot past the packed descriptors.
    if (tid == 0) {
        sTile = cuda::atomic_ref<uint, cuda::thread_scope_device>(
                    state[2u * numTiles])
                    .fetch_add(1u, cuda::memory_order_relaxed);
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

    scalar_t carry = (scalar_t)0;
    if (fullTile) {
        const VecT* __restrict__ src =
            reinterpret_cast<const VecT*>(dataIn + sliceStart);
        for (unsigned short i = 0; i < NVEC; i++) {
            v[i] = src[i];
            v[i].s[0] += carry;
            for (unsigned short j = 1; j < VW; j++) {
                v[i].s[j] += v[i].s[j - 1];
            }
            carry = v[i].s[VW - 1];
        }
    } else {
        for (unsigned short i = 0; i < NVEC; i++) {
            for (unsigned short j = 0; j < VW; j++) {
                const uint g = sliceStart + i * VW + j;
                v[i].s[j] = (g < pc.numElements) ? dataIn[g] : (scalar_t)0;
            }
            v[i].s[0] += carry;
            for (unsigned short j = 1; j < VW; j++) {
                v[i].s[j] += v[i].s[j - 1];
            }
            carry = v[i].s[VW - 1];
        }
    }

    scalar_t threadTotal = carry;

    // Block-wide inclusive scan of the per-thread totals (warp shuffle + shared).
    // cuda::device::warp_shuffle_up returns the shuffled value AND the predicate
    // the hardware already set for it — true exactly when the source lane is in
    // range, which at full warp width IS `lane >= off`. __shfl_up_sync has nowhere
    // to return that, so it forces a redundant compare and a select between the
    // shuffle and the add on every step.
    scalar_t x = threadTotal;
    for (unsigned short off = 1; off < 32; off <<= 1) {
        const auto up = cuda::device::warp_shuffle_up(x, off);
        if (up.pred) x += up.data;
    }
    if (lane == 31u) warpTotals[warp] = x;
    __syncthreads();

    // The NUM_WARPS totals stay a serial pass over shared, NOT the shuffle ladder
    // ScanDecoupled.cu uses. The ladder keeps the totals live in registers across
    // the look-back, and this kernel is register-limited by construction — that is
    // the whole point of the register-resident variant. Measured on sm_86: the
    // ladder costs +7 registers at IPT=32, +15 at 48, +29 at 64 (80 -> 109), which
    // drops IPT=64 from 3 resident blocks/SM to 2. ScanDecoupled.cu pays nothing
    // for it only because its 47 KB of staging pins it to 1 block/SM anyway, so
    // its register count is slack. Same code, opposite verdict.
    scalar_t warpPrefix = (scalar_t)0;
    scalar_t tileAgg = (scalar_t)0;
    for (unsigned short w = 0; w < NUM_WARPS; w++) {
        scalar_t s = warpTotals[w];
        tileAgg += s;
        if (w < warp) warpPrefix += s;
    }

    scalar_t threadPrefix = (x - threadTotal) + warpPrefix; // exclusive prefix of totals

    ull* const desc = DESC(state);

    // Publish this tile's aggregate BEFORE the fold, so a successor's look-back
    // is not blocked behind our fold. tileAgg is already known and the fold is
    // not on its critical path. (tile 0 has no predecessor aggregate to serve.)
    // Published from tid 32 — the first thread of warp 1 — rather than from tid 0,
    // which is the thread that then spins in the look-back below. Splitting them
    // across warps keeps the release store from sharing an issue slot with the
    // spin. tileAgg is block-uniform here, so any thread can publish it.
    if (tid == 32 && tile != 0u) {
        storeRelease64(&desc[tile], packDesc(FLAG_AGG, tileAgg));
    }

    // Fold only the per-thread part of the prefix into the registers now, while
    // tid 0's look-back below runs; the block-wide exclusive prefix is added at
    // the store instead of here, which is what lets the fold move ahead of the
    // look-back. The exclusive form shifts the slice right by one, so walk
    // downwards: a component's predecessor is still untouched when read.
    for (unsigned short i = NVEC; i-- > 0;) {
        for (unsigned short j = VW; j-- > 0;) {
            scalar_t prev;
            if (j > 0u)
                prev = v[i].s[j - 1];
            else if (i > 0u)
                prev = v[i - 1].s[VW - 1];
            else
                prev = (scalar_t)0;
            v[i].s[j] = (pc.isExclusive != 0u) ? (threadPrefix + prev)
                                              : (threadPrefix + v[i].s[j]);
        }
    }

    // Warp 0 walks the predecessors; ScanCommon.cuh carries the rationale for
    // the windowed, ready-prefix shape. `exclusive` comes back warp-uniform, so
    // tid 0 below has the value the INCLUSIVE publish needs. sExclusive is
    // written ONLY by tid 0 and the __syncthreads() that publishes it is
    // unconditional — the two invariants the shared-staged kernel got wrong.
    scalar_t exclusive = (scalar_t)0;
    if (tile > 0u && warp == 0u) {
        exclusive = scanWarpLookBack(desc, tile, lane);
    }
    if (tid == 0) {
        // Covers tile 0 too: exclusive is 0 there, so this is the same publish the
        // old `if (tile == 0u)` arm made.
        storeRelease64(&desc[tile], packDesc(FLAG_INC, exclusive + tileAgg));
        sExclusive = exclusive;
    }
    __syncthreads();
    const scalar_t tileExclusive = sExclusive;

    // Store the slice, adding the block-wide exclusive prefix the fold left out.
    if (fullTile) {
        VecT* __restrict__ dst = reinterpret_cast<VecT*>(dataOut + sliceStart);
        for (unsigned short i = 0; i < NVEC; i++) {
            for (unsigned short j = 0; j < VW; j++) {
                v[i].s[j] += tileExclusive;
            }
            dst[i] = v[i];
        }
    } else {
        for (unsigned short i = 0; i < NVEC; i++) {
            for (unsigned short j = 0; j < VW; j++) {
                const uint g = sliceStart + i * VW + j;
                if (g < pc.numElements) {
                    dataOut[g] = v[i].s[j] + tileExclusive;
                }
            }
        }
    }
}
