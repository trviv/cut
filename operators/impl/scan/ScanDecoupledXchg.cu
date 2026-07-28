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
#define XCHG_ROUNDS (NUM_WARPS / XW)
// One pad slot per lane slice; see the header. Keep the two views' index math in
// one place: XPHYS maps an element's position within the warp's run to its slot.
#define REGION (32 * (IPT + 1))
#define XPHYS(e) ((e) / IPT + (e))

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

// Packed per-tile descriptor plus its acquire/release accessors — the same
// protocol as ScanDecoupled.cu. cuda::atomic_ref at device scope emits
// ld.acquire.gpu / st.release.gpu on sm_70+ and handles older-arch lowering
// itself, so there is no __CUDA_ARCH__ dispatch to maintain here.
#define DESC(state) ((ull*)(state))
using DescRef = cuda::atomic_ref<ull, cuda::thread_scope_device>;

__device__ __forceinline__ ull packDesc(uint flag, scalar_t value) {
    return (ull)flag | ((ull)scalarToBits(value) << 32);
}
__device__ __forceinline__ uint descFlag(ull d) { return (uint)d; }
__device__ __forceinline__ scalar_t descValue(ull d) {
    return bitsToScalar((uint)(d >> 32));
}
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
    __shared__ scalar_t sXchg[XW * REGION];
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
    const uint warpStart = tileStart + (uint)warp * WARP_RUN;
    // Only the final tile can be partial; every other tile skips the per-element
    // bounds check on both the load and the store.
    const bool fullTile = (tileStart + TILE) <= pc.numElements;

    // Warp-striped load: lane l takes elements l, l+32, l+64, ... of this warp's
    // run, so each load instruction covers 32 consecutive elements. Every loop
    // that indexes v[] is #pragma unroll'd — the index has to be a compile-time
    // constant or ptxas puts the array in local memory (check: 0 spill bytes).
    scalar_t v[IPT];
    if (fullTile) {
        for (unsigned short i = 0; i < IPT; i++) {
            v[i] = __ldcs(&dataIn[warpStart + i * 32u + lane]);
        }
    } else {
        for (unsigned short i = 0; i < IPT; i++) {
            const uint g = warpStart + i * 32u + lane;
            v[i] = (g < pc.numElements) ? __ldcs(&dataIn[g]) : (scalar_t)0;
        }
    }

    // Striped -> blocked, warp-locally, XW warps at a time through one window.
    // The __syncthreads() is outside the round guard so the whole block reaches
    // it; it separates one group's readback from the next group's writes. Round 0
    // does not strictly need it on the load exchange (nothing has touched sXchg
    // yet) but keeping the loop uniform is worth one barrier.
    scalar_t* const region = sXchg + (warp % XW) * REGION;
    const unsigned short myRound = warp / XW;
    for (unsigned short k = 0; k < XCHG_ROUNDS; k++) {
        __syncthreads();
        if (myRound == k) {
            for (unsigned short i = 0; i < IPT; i++) {
                region[XPHYS(i * 32u + lane)] = v[i];
            }
            // Same warp writes and reads, so a warp-scoped barrier is enough —
            // but it IS required: independent thread scheduling does not
            // guarantee the write half has retired otherwise.
            __syncwarp();
            for (unsigned short r = 0; r < IPT; r++) {
                v[r] = region[lane * (IPT + 1) + r];
            }
        }
    }

    // v is now the blocked slice [tid*IPT, tid*IPT+IPT) — identical to what every
    // other variant holds at this point, so the rest of the kernel is the shared
    // algorithm. Scan the slice in place.
    for (unsigned short r = 1; r < IPT; r++) {
        v[r] += v[r - 1];
    }
    const scalar_t threadTotal = v[IPT - 1];

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

    const scalar_t threadPrefix = (x - threadTotal) + warpPrefix; // exclusive prefix of totals

    // Publish this tile's aggregate BEFORE the fold. tileAgg is known as soon as
    // the spine above finishes and every successor's look-back is blocked on it,
    // so the fold must not sit in front of it.
    ull* const desc = DESC(state);
    // Published from tid 32 — the first thread of warp 1 — rather than from tid 0,
    // which is the thread that then spins in the look-back below. Splitting them
    // across warps keeps the release store from sharing an issue slot with the
    // spin. tileAgg is block-uniform here, so any thread can publish it.
    if (tid == 32 && tile != 0u) {
        storeRelease64(&desc[tile], packDesc(FLAG_AGG, tileAgg));
    }

    // Fold ONLY the per-thread part of the prefix, while the slice is still
    // blocked. threadPrefix belongs to THIS thread's slice and can only be applied
    // on this side of the exchange below — afterwards a thread holds elements
    // owned by other threads, whose threadPrefix differs. The tile-wide exclusive
    // prefix is block-uniform, so it is the one term that may be added on the far
    // side of the transpose, which is what lets the look-back move after the fold.
    // The exclusive form shifts the slice right by one, so walk downwards: an
    // element's predecessor is still untouched when read.
    for (unsigned short r = IPT; r-- > 0;) {
        const scalar_t prev = (r > 0u) ? v[r - 1] : (scalar_t)0;
        v[r] = (pc.isExclusive != 0u) ? (threadPrefix + prev)
                                      : (threadPrefix + v[r]);
    }

    // Decoupled look-back on a single thread. NO barrier follows it: the store
    // exchange's round-0 __syncthreads() below is what publishes sExclusive to the
    // block. That merge is the saving — this kernel now synchronises XCHG_ROUNDS
    // times after the block scan instead of XCHG_ROUNDS + 1. It holds because
    // XCHG_ROUNDS = NUM_WARPS/XW is >= 1 for every XW the manifest carries; an XW
    // past NUM_WARPS would leave zero barriers and race the broadcast.
    scalar_t exclusive = (scalar_t)0;
    if (tid == 0) {
        if (tile == 0u) {
            storeRelease64(&desc[tile], packDesc(FLAG_INC, tileAgg));
        } else {
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
            // storeRelease64(&desc[tile], packDesc(FLAG_INC, exclusive + tileAgg));
        }
        sExclusive = exclusive;
    }

    // Blocked -> striped through the same window, so the stores are coalesced
    // exactly like the loads. Round 0's barrier carries two jobs now: it keeps a
    // warp's writes off a region another warp is still reading, and it is the
    // release point for sExclusive above.
    for (unsigned short k = 0; k < XCHG_ROUNDS; k++) {
        __syncthreads();
        if (myRound == k) {
            for (unsigned short r = 0; r < IPT; r++) {
                region[lane * (IPT + 1) + r] = v[r];
            }
            __syncwarp();
            for (unsigned short i = 0; i < IPT; i++) {
                v[i] = region[XPHYS(i * 32u + lane)];
            }
        }
    }

    if (warp == 0 && tid == 0) {
        storeRelease64(&desc[tile], packDesc(FLAG_INC, exclusive + tileAgg));
    }

    // Safe to read now: every thread has passed at least round 0's barrier.
    const scalar_t tileExclusive = sExclusive;

    if (fullTile) {
        for (unsigned short i = 0; i < IPT; i++) {
            __stcs(&dataOut[warpStart + i * 32u + lane], v[i] + tileExclusive);
        }
    } else {
        for (unsigned short i = 0; i < IPT; i++) {
            const uint g = warpStart + i * 32u + lane;
            if (g < pc.numElements) {
                __stcs(&dataOut[g], v[i] + tileExclusive);
            }
        }
    }
}
