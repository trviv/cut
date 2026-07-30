// Shared decoupled-look-back protocol for the native CUDA scan kernels.
//
// ScanDecoupled.cu, ScanDecoupledXchg.cu and ScanDecoupledReg.cu are three
// staging strategies over ONE algorithm; everything about the inter-tile
// protocol is identical between them and lives here. Only the global <-> compute
// data path differs, and that stays in each .cu:
//
//   ScanDecoupled      global --striped--> registers --> __shared__ sData[TILE]
//   ScanDecoupledXchg  global --striped--> registers                 (no staging)
//   ScanDecoupledReg   global --vec4-----> registers                 (no staging)
//
// The Vulkan .shader counterparts keep their own copy of the protocol because
// HLSL cannot express the 64-bit packed descriptor (no 64-bit atomics at cs_6_2 /
// vulkan1.1) — see the note in ScanDecoupled.shader.
//
// STATE LAYOUT. Each tile owns ONE 64-bit descriptor, packed as
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
// counter therefore lands at state[2*T], the first slot past them.
//
// NOTE: decoupled look-back relies on concurrent block forward progress. The
// dynamic tile counter makes claim order match launch order so a spinning tile
// only ever waits on tiles that were claimed earlier; validated on the target
// GPU rather than portable to every scheduler.
#pragma once

#include <cuda/atomic>

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
// there is nothing left for a __threadfence() to order. Relaxed is NOT enough
// either: it drops both MEMBAR.ALL.GPU from the SASS and fails scan_verify.
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

// Warp-parallel decoupled look-back. Returns this tile's exclusive prefix,
// WARP-UNIFORM across the calling warp.
//
// CONTRACT: call with all 32 lanes of ONE warp converged, and only when
// tile > 0. `lane` must be the caller's lane id.
//
// Warp 0 probes a WINDOW of 32 predecessors per step instead of walking them one
// at a time, so the walk costs ceil(depth/32) dependent L2 round trips rather
// than `depth` of them. That matters because the serial walk is quadratic in the
// worst case that actually occurs: when every tile is co-resident they all
// publish AGGREGATE at about the same instant, and tile t then has to sum its way
// back toward the INCLUSIVE frontier one 200-cycle load at a time while t+1 does
// the same one slot behind.
//
// The trade is traffic for latency: a window always issues 32 loads even when the
// immediate predecessor is already INCLUSIVE and one would have done. Descriptors
// are 8 B and L2-resident, so that is cheap next to a serialized dependent chain.
__device__ __forceinline__ scalar_t
scanWarpLookBack(ull *desc, uint tile, unsigned int lane) {
    const uint FULL = 0xffffffffu;
    scalar_t exclusive = (scalar_t)0;
    int base = (int)tile - 1;  // lane l probes descriptor base - l

    for (;;) {
        const int p = base - (int)lane;
        // Lanes below tile 0 act as a zero-valued INCLUSIVE terminator, which
        // stops the walk without any lane having to probe.
        unsigned short flag = FLAG_INC;
        scalar_t val = (scalar_t)0;
        if (p >= 0) {
            // One acquire load yields both halves, so the value needs no second
            // dependent load and no fence to pair it with the flag.
            const ull d = loadAcquire64(&desc[(uint)p]);
            flag = descFlag(d);
            val = descValue(d);
        }

        // NOTHING SPINS PER LANE, and that is the whole design. Spinning each lane
        // until its own descriptor is valid — the obvious form, and the one CUB
        // uses — makes a window cost the SLOWEST of its 32 predecessors, while the
        // serial walk it replaces only ever waited on the nearest. Measured, that
        // inversion costs 4-6% at 16M (where the immediate predecessor is usually
        // INCLUSIVE already) and eats much of the win at 1M.
        //
        // Instead, take only the READY PREFIX of the window. Predecessors must be
        // consumed contiguously, so the first NOT_READY lane bounds what this pass
        // may claim; everything before it is summed now and the window slides by
        // exactly that much. A stalled far predecessor is simply never waited on,
        // and re-probing costs one more window, not one more element.
        const uint readyMask = __ballot_sync(FULL, flag != 0u);
        const uint notReady = ~readyMask;
        const unsigned short firstNR = (notReady != 0u) ? (__ffs((int)notReady) - 1u) : 32u;
        const uint usable = (firstNR >= 32) ? FULL : ((1u << firstNR) - 1u);
        // Nearest INCLUSIVE inside the ready prefix terminates the sum: it already
        // carries everything below it. __ffs finds the lowest set bit == the
        // nearest tile, since lane l holds the l-th predecessor.
        const uint incMask = __ballot_sync(FULL, flag == FLAG_INC) & usable;
        const bool done = (incMask != 0u);
        const short stop = (done ? __ffs((int)incMask) : firstNR) - 1;
        if ((int)lane > stop) {
            val = (scalar_t)0;
        }

        // Butterfly rather than shuffle-down so every lane ends up with the window
        // total and `exclusive` stays warp-uniform — the loop exit is uniform too,
        // so no lane can fall out mid-accumulation.
        for (unsigned short off = 16; off > 0; off >>= 1) {
            val += __shfl_xor_sync(FULL, val, off);
        }
        exclusive += val;

        if (done) {
            break;
        }
        // firstNR == 0 means even the nearest predecessor is still NOT_READY:
        // nothing was claimed and `base` must not move, which is exactly the
        // serial walk's spin — one window wide.
        base -= firstNR;
    }
    return exclusive;
}
