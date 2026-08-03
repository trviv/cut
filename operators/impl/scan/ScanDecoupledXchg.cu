// Native CUDA counterpart of ScanDecoupledXchg.shader — keep semantics in lockstep.
//
// Third staging strategy for the decoupled look-back scan. Same single-pass
// algorithm, same tile geometry (TILE = BLOCK*IPT), same packed descriptor
// protocol as ScanDecoupled.cu / ScanDecoupledReg.cu — the ONLY difference is
// how a tile gets between global memory and each thread's blocked IPT-element
// slice:
//
//   ScanDecoupled      global --striped--> __shared__ sData[TILE] --> registers
//   ScanDecoupledReg   global --vec4-----> registers                 (no staging)
//   ScanDecoupledXchg  global --striped--> registers --> __shared__ window --> registers
//
// ScanDecoupled buys a perfectly coalesced global load with BLOCK*IPT*4 bytes of
// shared — the whole tile — which pins the larger IPT variants to one block/SM.
// ScanDecoupledReg hands that shared memory back and pays at the load instead: a
// warp's 32 lanes each read their own contiguous slice, so every load spans 32
// cache lines. This variant keeps the coalesced load AND most of the shared
// memory, by never making the whole tile resident at once.
//
// The trick is WHICH striped mapping to load. A block-striped load (thread t
// takes elements t, t+BLOCK, t+2*BLOCK, ... — what ScanDecoupled does) cannot be
// transposed through a window: a thread's blocked slice is assembled from
// registers held by every thread in the block, so a window-sized buffer can only
// serve the few threads whose slice lands inside the current window, and those
// threads' own registers are overwritten before they have donated the elements
// belonging to later windows. Fixing that costs a second v[IPT] array (2x the
// registers, the one resource this family is already short on) or a re-read of
// global memory per window.
//
// A WARP-striped load has no such problem. Warp w takes the contiguous run
// [tileStart + w*32*IPT, +32*IPT), lane l taking elements i*32 + l of it — still
// 32 consecutive elements per load instruction, so still perfectly coalesced.
// But that run is exactly the run w's own lanes need after the transpose, since
// lane l's blocked slice is [w*32*IPT + l*IPT, +IPT) = [tid*IPT, +IPT), the same
// slice every other variant gives it. So the exchange is warp-local: nothing
// crosses a warp boundary, one warp's run (32*IPT elements) is the entire
// working set, and a warp writes its whole slice before reading any of it back.
// Warps then take turns through that one buffer, XW at a time.
//
//   shared = XW * 32*(IPT+1) elements   (XW=1: TILE/8 + padding)
//   barriers = NUM_WARPS/XW per exchange, two exchanges (load and store) per tile
//
// So XW trades shared memory against barrier count and idle warps: at XW=1 seven
// of eight warps sit at the barrier during each round, at XW=8 there are no
// rounds at all and the buffer is the whole tile again (ScanDecoupled, reached by
// a different road). Where the optimum sits is a measurement, not a derivation —
// hence a variant family with XW in the manifest.
//
// The +1 per lane slice is the bank-conflict pad. The blocked readback has lane l
// reading a run based at l*IPT, so an unpadded buffer collides gcd(IPT,32) ways —
// 32 ways at IPT=32 or 64. One pad slot per lane slice makes the stride IPT+1,
// odd for every even IPT, so the readback is conflict-free; the striped write is
// conflict-free with or without it (consecutive lanes, consecutive addresses).
//
// Because shared no longer scales with the tile, this template also carries IPT
// values (48, 64) that ScanDecoupled cannot express on a 48 KB device.
//
// Element type is the generator's CUT_SCALAR_DTYPE_INPUT (Float32 / Int32 /
// UInt32); look-back descriptors store the value's bit pattern in a uint slot.
//
// State layout — identical to ScanDecoupled.cu, which see for the rationale.
// Each tile owns ONE 64-bit descriptor: low 32 bits the status flag (0 =
// NOT_READY, 1 = AGGREGATE, 2 = INCLUSIVE), high 32 bits the matching value's
// bits, so a look-back step gets both halves in a single acquire load.
// Descriptors occupy state[0 .. 2*T) viewed as ull and the dynamic tile counter
// lands at state[2*T] — 2*T + 1 uints, which is what ScanOp.cpp allocates.
//
// NOTE: decoupled look-back relies on concurrent block forward progress. The
// dynamic tile counter makes claim order match launch order so a spinning tile
// only ever waits on tiles that were claimed earlier; validated on the target
// GPU rather than portable to every scheduler.
// NOTE (warp-cooperative rewrite): the striped -> blocked exchange this file was
// built around is GONE. The scan runs directly on the striped registers with a
// warp shuffle ladder and a running carry, so nothing needs a blocked slice and
// nothing needs the shared window. XW therefore has NO effect on the generated
// code — every XchgW*IPT<N> variant with the same BLOCK and IPT now compiles to
// an identical kernel, and the XW axis of the manifest is redundant.
#include "ComputeOpsShared.h"
#include <cuda/atomic>
#include <cuda/warp>

#ifndef BLOCK
#define BLOCK 256  // overridden per-variant by the native manifest (-DBLOCK=N)
#endif
#ifndef IPT
#define IPT 32  // overridden per-variant by the native manifest (-DIPT=N)
#endif
#ifndef XW
#define XW 1  // warps sharing the exchange buffer per round (-DXW=N)
#endif
#define TILE (BLOCK * IPT)      // elements per tile (8192 at IPT=32)
#define NUM_WARPS (BLOCK / 32)  // 8
#define WARP_RUN (32 * IPT)     // elements one warp loads, and exchanges
// One pad slot per lane slice; see the header. Keep the two views' index math in
// one place: XPHYS maps an element's position within the warp's run to its slot.

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
    const uint threadStart = tileStart + (uint)warp * WARP_RUN + lane;
    // Only the final tile can be partial; every other tile skips the per-element
    // bounds check on both the load and the store.
    const bool fullTile = (tileStart + TILE) <= pc.numElements;

    // Warp-striped load: lane l takes elements l, l+32, l+64, ... of this warp's
    // run, so each load instruction covers 32 consecutive elements. Every loop
    // that indexes v[] is #pragma unroll'd — the index has to be a compile-time
    // constant or ptxas puts the array in local memory (check: 0 spill bytes).
    //
    // These pragmas were silently LOST in the warp-cooperative rewrite (05a01f9),
    // leaving only the comment above claiming they were there. ptxas still
    // unrolled IPT<=48 on its own so nothing looked broken, but at IPT=64 it left
    // the scan loop rolled — 7 SHFL for the whole kernel instead of ~390 — and
    // every IPT64 variant ran at 273-277 GB/s against ~780 for its neighbours.
    // Restoring them is worth 2.1x at 4M and 2.7x at 16M for those variants. If a
    // variant ever posts a number wildly out of line with its neighbours, check
    // the SASS for a rolled loop before anything else.
    scalar_t v[IPT];
    if (fullTile) {
        #pragma unroll
        for (unsigned short i = 0; i < IPT; i++) {
            v[i] = __ldcs(&dataIn[threadStart + i * 32u]);
        }
    } else {
        #pragma unroll
        for (unsigned short i = 0; i < IPT; i++) {
            const uint g = threadStart + i * 32u;
            v[i] = (g < pc.numElements) ? __ldcs(&dataIn[g]) : (scalar_t)0;
        }
    }

    // Warp-cooperative scan over the run, 32 elements at a time — the shape
    // ScanDecoupled.cu moved to. v[i] already holds 32 CONSECUTIVE elements across
    // the warp (lane l owns element i*32+l of the run), which is exactly what a
    // warp scan wants, so the striped -> blocked exchange that used to sit here
    // has no reason to exist: nothing downstream needs a blocked slice. That
    // removes sXchg, both transposes and their 2*XCHG_ROUNDS barriers.
    //
    // #pragma unroll is NOT optional: the inner shuffle ladder stops ptxas
    // unrolling this loop on its own, and a runtime index into v[] puts the whole
    // array in local memory. Measured without it: 3120us at N=16M against 175us
    // with it, an 18x cliff.
    scalar_t carry = (scalar_t)0;

    #pragma unroll
    for (unsigned short i = 0; i < IPT; i++) {
        const scalar_t orig = v[i];
        scalar_t x = orig;
        for (unsigned short off = 1; off < 32; off <<= 1) {
            const auto up = cuda::device::warp_shuffle_up(x, off);
            if (up.pred) x += up.data;
        }
        x += carry;
        // Run-local result; the run's prefix and the tile's are both warp-uniform
        // and are added once, at the store.
        v[i] = (pc.isExclusive != 0u) ? (x - orig) : x;
        carry = cuda::device::warp_shuffle_idx(x, 31);
    }

    if (lane == 0u) warpTotals[warp] = carry;  // this run's total
    __syncthreads();

    // The NUM_WARPS totals stay a serial pass over shared, NOT the shuffle ladder
    // ScanDecoupled.cu uses, for the reason ScanDecoupledReg.cu spells out: the
    // ladder keeps the totals live in registers across the look-back, and this
    // kernel is register-limited by construction. Shrinking the staging buffer is
    // the entire point of the variant, so shared memory is exactly the resource
    // with slack here and registers are the one without.
    scalar_t warpPrefix = (scalar_t)0;
    scalar_t tileAgg = (scalar_t)0;
    for (unsigned short w = 0; w < NUM_WARPS; w++) {
        const scalar_t s = warpTotals[w];
        tileAgg += s;
        if (w < warp) warpPrefix += s;
    }

    // Publish this tile's aggregate as soon as the spine finishes: tileAgg is known then and
    // the spine above finishes and every successor's look-back is blocked on it,
    // so the fold must not sit in front of it.
    ull* const desc = DESC(state);

    // Warp-parallel decoupled look-back — same shape as ScanDecoupled.cu, which
    // carries the full rationale. Warp 0 probes a WINDOW of 32 predecessors per
    // step so the walk costs ceil(depth/32) dependent L2 round trips instead of
    // `depth` of them, and takes only the window's READY PREFIX so it never waits
    // on a far predecessor the serial walk would not have waited on.
    scalar_t prevBlockExclusive = (scalar_t)0;
    scalar_t exclusive = (scalar_t)0;
    if (tile > 0u) {
        if (tid == 32) {
            storeRelease64(&desc[tile], packDesc(FLAG_AGG, tileAgg));
        }
        if (warp == 0u) {
            exclusive = scanWarpLookBack(desc, tile, lane);
            if (lane == 0u)
                sExclusive = exclusive;
        }
        __syncthreads();
        prevBlockExclusive = sExclusive;
    }
    if (tid == 0) {
        storeRelease64(&desc[tile], packDesc(FLAG_INC, exclusive + tileAgg));
    }

    // Safe to read now: the barrier above ordered the look-back's write.
    const scalar_t bias = warpPrefix + prevBlockExclusive;

    if (fullTile) {
        #pragma unroll
        for (uint i = 0; i < IPT; i++) {
            __stcs(&dataOut[threadStart + i * 32u], v[i] + bias);
        }
    } else {
        #pragma unroll
        for (uint i = 0; i < IPT; i++) {
            const uint g = threadStart + i * 32u;
            if (g < pc.numElements) {
                __stcs(&dataOut[g], v[i] + bias);
            }
        }
    }
}
