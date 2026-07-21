// Native CUDA counterpart of OneSweepScatter.shader. One OneSweep digit pass
// with decoupled look-back, using LARGE tiles (KPT elements per thread) plus a
// light warp-cooperative rank, to cut the tile count (shorter look-back chains,
// smaller grid) and produce coalesced global writes at scale.
//
//   * One block == one tile of TILE = BLOCK*KPT elements; tile id claimed from a
//     global atomic counter (NOT blockIdx) for forward-progress order.
//   * The tile is ranked in KPT "rounds" of BLOCK elements. Each round ranks its
//     elements by digit with __match_any_sync + __popc (intra-warp) plus a
//     per-warp histogram scan (inter-warp), and a running per-digit count keeps
//     the rank stable across rounds — no block-wide bit-split scans.
//   * Thread `tid` owns digit `tid`: publishes this tile's aggregate, walks
//     predecessors to an INCLUSIVE descriptor, publishes the inclusive prefix.
//   * The tile is locally sorted into shared memory, so equal-digit elements are
//     drained to contiguous global slots: pos = globalHist[pass*RADIX+d]
//     + exclusivePrefix[d] + (sortedPos - digitBase[d]).
//
// Partial last tile: out-of-range slots use key 0xFFFFFFFF (digit 0xFF every
// pass) and are excluded from the histogram / never written. Descriptor packing:
// value bits [29:0], status bits [31:30] (0=NOT_READY,1=AGGREGATE,2=INCLUSIVE);
// value width => numElements < 2^30. Each pass owns a contiguous look-back
// region (stateBase); the trailing slot is that pass's tile counter. Requires
// SM 7.0+ for __match_any_sync (Ampere target).
#include "ComputeOpsShared.h"

#define BLOCK 256
#define RADIX 256
#define KPT 4
#define TILE (BLOCK * KPT)          // 1024 elements per tile
#define NUM_WARPS (BLOCK / 32)      // 8
#define FLAG_AGG (1u << 30)
#define FLAG_INC (2u << 30)
#define FLAG_MASK (3u << 30)
#define VALUE_MASK 0x3FFFFFFFu

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
    __shared__ uint sKey[TILE];
    __shared__ uint sVal[TILE];
    __shared__ uint sRank[TILE];               // stable tile-rank per position
    __shared__ uint sSorted[TILE];             // sorted pos -> original position
    __shared__ uint warpHist[NUM_WARPS * RADIX];
    __shared__ uint sHist[RADIX];              // running/final per-digit count
    __shared__ uint sBase[RADIX];              // exclusive scan of sHist
    __shared__ uint sExclusive[RADIX];         // look-back exclusive prefix
    __shared__ uint sTile;

    uint tid = threadIdx.x;
    uint lane = tid & 31u;
    uint warpId = tid >> 5u;

    uint stateStride = pc.numTiles * RADIX + 1u;
    uint stateBase = pc.passIndex * stateStride;

    if (tid == 0) {
        sTile = atomicAdd(&lookbackState[stateBase + pc.numTiles * RADIX], 1u);
    }
    __syncthreads();
    uint tile = sTile;

    uint bitOffset = pc.passIndex * 8u;
    uint tileStart = tile * TILE;

    if (tid < RADIX) {
        sHist[tid] = 0u;
    }
    __syncthreads();

    // Coalesced striped load into natural tile-position order.
#pragma unroll
    for (uint r = 0; r < KPT; r++) {
        uint i = r * BLOCK + tid;
        uint g = tileStart + i;
        sKey[i] = (g < pc.numElements) ? keysIn[g] : 0xFFFFFFFFu;
        sVal[i] = (g < pc.numElements) ? valsIn[g] : 0u;
    }
    __syncthreads();

    // KPT rounds: stable within-tile rank per element; accumulate the histogram.
    for (uint r = 0; r < KPT; r++) {
        for (uint i = tid; i < NUM_WARPS * RADIX; i += BLOCK) {
            warpHist[i] = 0u;
        }
        __syncthreads();

        uint i = r * BLOCK + tid;
        uint g = tileStart + i;
        uint valid = (g < pc.numElements) ? 1u : 0u;
        uint digit = valid ? ((sKey[i] >> bitOffset) & 0xFFu) : 0xFFFFFFFFu;

        uint match = __match_any_sync(__activemask(), digit);
        uint intra = __popc(match & ((1u << lane) - 1u));
        if (valid && intra == 0u) {
            warpHist[warpId * RADIX + digit] = __popc(match);
        }
        __syncthreads();

        if (valid) {
            uint inter = 0u;
            for (uint w = 0; w < warpId; w++) {
                inter += warpHist[w * RADIX + digit];
            }
            sRank[i] = sHist[digit] + inter + intra;
        }
        __syncthreads();

        // Thread `tid` owns digit tid: fold this round's count into the running
        // total (read by the next round's rank).
        uint roundCount = 0u;
        for (uint w = 0; w < NUM_WARPS; w++) {
            roundCount += warpHist[w * RADIX + tid];
        }
        sHist[tid] += roundCount;
        __syncthreads();
    }

    // Exclusive scan of the histogram -> local digit base; capture total valid.
    sBase[tid] = sHist[tid];
    __syncthreads();
    for (uint off = 1u; off < RADIX; off <<= 1) {
        uint add = (tid >= off) ? sBase[tid - off] : 0u;
        __syncthreads();
        sBase[tid] += add;
        __syncthreads();
    }
    uint totalValid = sBase[RADIX - 1];  // inclusive scan total
    __syncthreads();
    sBase[tid] = sBase[tid] - sHist[tid];  // exclusive
    __syncthreads();

    // Local sort: place each valid element at its sorted position in shared.
#pragma unroll
    for (uint r = 0; r < KPT; r++) {
        uint i = r * BLOCK + tid;
        uint g = tileStart + i;
        if (g < pc.numElements) {
            uint dd = (sKey[i] >> bitOffset) & 0xFFu;
            sSorted[sBase[dd] + sRank[i]] = i;
        }
    }
    __syncthreads();

    // Decoupled look-back: thread d == tid owns digit d.
    uint d = tid;
    uint agg = sHist[d];
    if (tile == 0u) {
        atomicExch(&lookbackState[stateBase + tile * RADIX + d],
                   (agg & VALUE_MASK) | FLAG_INC);
        sExclusive[d] = 0u;
    } else {
        atomicExch(&lookbackState[stateBase + tile * RADIX + d],
                   (agg & VALUE_MASK) | FLAG_AGG);
        uint exclusive = 0u;
        int pred = (int)tile - 1;
        while (pred >= 0) {
            uint w;
            do {
                w = atomicAdd(&lookbackState[stateBase + (uint)pred * RADIX + d], 0u);
            } while ((w & FLAG_MASK) == 0u);
            exclusive += (w & VALUE_MASK);
            if ((w & FLAG_MASK) == FLAG_INC) {
                break;
            }
            pred--;
        }
        atomicExch(&lookbackState[stateBase + tile * RADIX + d],
                   ((exclusive + agg) & VALUE_MASK) | FLAG_INC);
        sExclusive[d] = exclusive;
    }
    __syncthreads();

    // Drain the locally-sorted tile to global (coalesced within each digit).
    uint globalBase = pc.passIndex * RADIX;
#pragma unroll
    for (uint r = 0; r < KPT; r++) {
        uint p = r * BLOCK + tid;
        if (p < totalValid) {
            uint elem = sSorted[p];
            uint key = sKey[elem];
            uint dd = (key >> bitOffset) & 0xFFu;
            uint pos = globalHist[globalBase + dd] + sExclusive[dd] + (p - sBase[dd]);
            keysOut[pos] = key;
            valsOut[pos] = sVal[elem];
        }
    }
}
