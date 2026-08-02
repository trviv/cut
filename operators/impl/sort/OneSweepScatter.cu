// Native CUDA counterpart of OneSweepScatter.shader. One OneSweep digit pass
// with decoupled look-back over LARGE tiles (KPT elements per thread), ranked
// with warp match intrinsics and exchanged through shared memory so the global
// writes stay coalesced.
//
//   * One block == one tile of TILE = BLOCK*KPT elements; the tile id is claimed
//     from a global atomic counter (NOT blockIdx) so the look-back chain has a
//     forward-progress order.
//   * WARP-STRIPED load. Lane l of warp w owns tile positions
//     w*32*KPT + r*32 + l. Consecutive lanes read consecutive addresses, so the
//     load is coalesced, and iterating r-major then lane visits a warp's
//     positions in increasing order while each warp owns a contiguous range —
//     which is exactly the order the rank below counts in, and therefore what
//     makes the rank STABLE. Every radix pass after the first is wrong without
//     that. (This is CUB's BLOCK_LOAD_WARP_TRANSPOSE, same as the scan.)
//   * RANK. Each warp keeps its own RADIX counters and ranks its items with
//     __match_any_sync + __popc: the lowest lane of each same-digit peer group
//     reads and bumps the counter once for the whole group. No block barrier and
//     no bit-split scan anywhere in the ranking loop.
//   * Thread `d` owns digit d: it folds the per-warp counts into the tile
//     aggregate, publishes it, walks predecessors to an INCLUSIVE descriptor and
//     publishes this tile's inclusive prefix.
//   * The tile is sorted into shared memory and drained in sorted order, so
//     equal-digit runs land in contiguous global slots:
//       pos = globalHist[pass*RADIX+d] + exclusivePrefix[d] + (p - digitBase[d])
//     which is precomputed per digit as sGlobalBase[d] + p. Keys and values go
//     through the same shared buffer in two phases rather than two buffers,
//     because shared memory is what caps occupancy here.
//
// Partial last tile: out-of-range slots are excluded from the histogram, the
// rank and the drain, and are matched against a sentinel that no real digit can
// collide with. Descriptor packing: value bits [29:0], status bits [31:30]
// (0=NOT_READY, 1=AGGREGATE, 2=INCLUSIVE); the value width caps numElements at
// 2^30. Each pass owns a contiguous look-back region; the trailing slot is that
// pass's tile counter. Requires SM 7.0+ for __match_any_sync (Ampere target).
//
// KPT is duplicated in SortOp.cpp (buildOneSweepGraph picks elemsPerTile from
// it) and the two MUST agree: the kernel derives its tile from a compile-time
// TILE while the host derives the grid and the look-back state size from the
// same number.
#include "ComputeOpsShared.h"

#include <cuda/atomic>

#define BLOCK 256
#define RADIX 256
#define KPT 12
#define TILE (BLOCK * KPT)
#define NUM_WARPS (BLOCK / 32)
#define FLAG_AGG (1u << 30)
#define FLAG_INC (2u << 30)
#define FLAG_MASK (3u << 30)
#define VALUE_MASK 0x3FFFFFFFu

// Look-back descriptor traffic, same rationale as ScanCommon.cuh: the protocol
// needs a coherent, ordered view of other blocks' descriptors, and atomicAdd(p,0)
// / atomicExch buy that as a read-modify-write at the L2 atomic unit — far
// heavier than the scoped load/store that is actually required. The descriptor is
// one 32-bit word carrying both the flag and the value it vouches for, so a
// single acquire load is enough and there is nothing left for a __threadfence()
// to order.
using StateRef = cuda::atomic_ref<uint, cuda::thread_scope_device>;

struct PushConstants {
    uint numElements;
    uint passIndex;
    uint numTiles;
};

extern "C" __global__ void cut_main(const uint* __restrict__ keysIn,
                                    const uint* __restrict__ valsIn,
                                    uint* __restrict__ keysOut,
                                    uint* __restrict__ valsOut,
                                    const uint* __restrict__ globalHist,
                                    uint* __restrict__ lookbackState,
                                    PushConstants pc) {
    // Per-warp digit counters; rewritten in place as per-(warp, digit) tile
    // offsets once the block totals are known.
    __shared__ uint warpHist[NUM_WARPS * RADIX];
    // The locally-sorted tile. Holds keys first, then values — one buffer, two
    // phases, because this array is what decides how many blocks fit on an SM.
    __shared__ uint sExchange[TILE];
    __shared__ uint sGlobalBase[RADIX];
    __shared__ uint sWarpScan[NUM_WARPS];
    __shared__ uint sTile;

    const uint tid = threadIdx.x;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    const uint laneMaskLt = (1u << lane) - 1u;

    const uint stateStride = pc.numTiles * RADIX + 1u;
    uint *state = lookbackState + pc.passIndex * stateStride;

    if (tid == 0u) {
        sTile = StateRef(state[pc.numTiles * RADIX])
                    .fetch_add(1u, cuda::memory_order_relaxed);
    }
    // Zeroing the counters is independent of which tile we drew, so it fills the
    // atomic's latency instead of waiting behind it.
    for (uint i = tid; i < NUM_WARPS * RADIX; i += BLOCK) {
        warpHist[i] = 0u;
    }
    __syncthreads();

    const uint tile = sTile;
    const uint bitOffset = pc.passIndex * 8u;
    const uint tileStart = tile * TILE;
    const uint validCount =
        (tileStart >= pc.numElements) ? 0u
                                      : min((uint)TILE, pc.numElements - tileStart);

    // 1. Warp-striped load.
    const uint regionBase = warp * (32u * KPT);
    uint keys[KPT];
    uint vals[KPT];
#pragma unroll
    for (uint r = 0; r < KPT; r++) {
        const uint i = regionBase + r * 32u + lane;
        const uint g = tileStart + i;
        keys[r] = (i < validCount) ? keysIn[g] : 0xFFFFFFFFu;
        vals[r] = (i < validCount) ? valsIn[g] : 0u;
    }

    // 2. Warp-local match rank. `rank[r]` counts same-digit items that this warp
    //    has already placed, so it is the item's offset within (warp, digit).
    uint rank[KPT];
#pragma unroll
    for (uint r = 0; r < KPT; r++) {
        const uint i = regionBase + r * 32u + lane;
        const bool valid = (i < validCount);
        const uint digit = (keys[r] >> bitOffset) & 0xFFu;
        // 0x100 is outside the digit range, so out-of-range slots form their own
        // peer group and can never be mistaken for real 0xFF digits (which a
        // 0xFFFFFFFF pad key would otherwise collide with).
        const uint matchVal = valid ? digit : 0x100u;
        const uint peers = __match_any_sync(0xFFFFFFFFu, matchVal);
        const uint before = __popc(peers & laneMaskLt);
        uint base = 0u;
        if (valid && before == 0u) { // lowest lane of a valid peer group
            base = warpHist[warp * RADIX + digit];
            warpHist[warp * RADIX + digit] = base + __popc(peers);
        }
        base = __shfl_sync(0xFFFFFFFFu, base, __ffs(peers) - 1);
        rank[r] = base + before;
        __syncwarp();
    }
    __syncthreads();

    // 3. Thread `d` owns digit d from here on: fold the per-warp counts into the
    //    tile total, then exclusive-scan the RADIX totals to get each digit's
    //    base within the tile.
    const uint d = tid;
    uint counts[NUM_WARPS];
    uint digitTotal = 0u;
#pragma unroll
    for (uint w = 0; w < NUM_WARPS; w++) {
        counts[w] = warpHist[w * RADIX + d];
        digitTotal += counts[w];
    }

    uint x = digitTotal;
#pragma unroll
    for (uint off = 1u; off < 32u; off <<= 1) {
        const uint y = __shfl_up_sync(0xFFFFFFFFu, x, off);
        if (lane >= off)
            x += y;
    }
    if (lane == 31u)
        sWarpScan[warp] = x;
    __syncthreads();
    if (warp == 0u) {
        uint w = (lane < NUM_WARPS) ? sWarpScan[lane] : 0u;
#pragma unroll
        for (uint off = 1u; off < NUM_WARPS; off <<= 1) {
            const uint y = __shfl_up_sync(0xFFFFFFFFu, w, off);
            if (lane >= off)
                w += y;
        }
        if (lane < NUM_WARPS)
            sWarpScan[lane] = w;
    }
    __syncthreads();
    const uint digitBase =
        (warp == 0u ? 0u : sWarpScan[warp - 1u]) + x - digitTotal;

    // 4. Publish the aggregate before doing any data movement: a successor
    //    spinning on this tile can start summing while this block is still
    //    shuffling its own keys around.
    if (tile == 0u) {
        StateRef(state[d]).store((digitTotal & VALUE_MASK) | FLAG_INC,
                                 cuda::memory_order_release);
    } else {
        StateRef(state[tile * RADIX + d])
            .store((digitTotal & VALUE_MASK) | FLAG_AGG,
                   cuda::memory_order_release);
    }

    // 5. Rewrite the counters as tile-local start offsets for (warp, digit).
    uint running = digitBase;
#pragma unroll
    for (uint w = 0; w < NUM_WARPS; w++) {
        warpHist[w * RADIX + d] = running;
        running += counts[w];
    }
    __syncthreads();

    // 6. Sort the tile into shared memory.
    uint pos[KPT];
#pragma unroll
    for (uint r = 0; r < KPT; r++) {
        const uint i = regionBase + r * 32u + lane;
        if (i < validCount) {
            const uint digit = (keys[r] >> bitOffset) & 0xFFu;
            pos[r] = warpHist[warp * RADIX + digit] + rank[r];
            sExchange[pos[r]] = keys[r];
        }
    }

    // 7. Decoupled look-back. Lanes of a warp probe consecutive digits of the
    //    same predecessor tile, so each step of the walk is one coalesced
    //    transaction per warp rather than 32 scattered ones.
    uint exclusive = 0u;
    if (tile > 0u) {
        int pred = (int)tile - 1;
        while (pred >= 0) {
            uint w;
            do {
                w = StateRef(state[(uint)pred * RADIX + d])
                        .load(cuda::memory_order_acquire);
            } while ((w & FLAG_MASK) == 0u);
            exclusive += (w & VALUE_MASK);
            if ((w & FLAG_MASK) == FLAG_INC)
                break;
            pred--;
        }
        StateRef(state[tile * RADIX + d])
            .store(((exclusive + digitTotal) & VALUE_MASK) | FLAG_INC,
                   cuda::memory_order_release);
    }

    // Fold the three per-digit terms into one, so the drain below is a single
    // shared lookup plus an add. The subtraction can wrap; the drain adds
    // p >= digitBase back, so the modular arithmetic lands on the right slot.
    sGlobalBase[d] =
        globalHist[pc.passIndex * RADIX + d] + exclusive - digitBase;
    __syncthreads();

    // 8. Drain the keys in sorted order: consecutive threads take consecutive
    //    sorted positions, which are contiguous in global within each digit run.
    uint outPos[KPT];
#pragma unroll
    for (uint r = 0; r < KPT; r++) {
        const uint p = r * BLOCK + tid;
        if (p < validCount) {
            const uint key = sExchange[p];
            outPos[r] = sGlobalBase[(key >> bitOffset) & 0xFFu] + p;
            keysOut[outPos[r]] = key;
        }
    }
    __syncthreads();

    // 9. Same trip for the values, reusing the buffer and the positions both
    //    phases already computed.
#pragma unroll
    for (uint r = 0; r < KPT; r++) {
        const uint i = regionBase + r * 32u + lane;
        if (i < validCount) {
            sExchange[pos[r]] = vals[r];
        }
    }
    __syncthreads();
#pragma unroll
    for (uint r = 0; r < KPT; r++) {
        const uint p = r * BLOCK + tid;
        if (p < validCount) {
            valsOut[outPos[r]] = sExchange[p];
        }
    }
}
