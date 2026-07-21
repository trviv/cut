// Native CUDA counterpart of OneSweepScatter.shader. One OneSweep digit pass
// with decoupled look-back:
//
//   * One block == one tile of BLOCK elements (one element per thread).
//   * The tile id is claimed from a global atomic counter (NOT blockIdx) so the
//     chained look-back has a forward-progress guarantee.
//   * Thread `tid` owns digit `tid`: it publishes this tile's per-digit
//     aggregate, walks predecessor tiles summing their aggregates until it hits
//     an INCLUSIVE descriptor, then publishes this tile's inclusive prefix.
//   * Each element's global position is
//       globalHist[pass*RADIX + digit]   (global base for the digit)
//     + exclusivePrefix[digit]           (same-digit elements in earlier tiles)
//     + localRank                        (stable rank within this tile).
//
// Descriptor packing (lookbackState[pass_region + tile*RADIX + digit]): value in
// bits [29:0], status flag in bits [31:30] (0=NOT_READY, 1=AGGREGATE,
// 2=INCLUSIVE). Publishes use atomicExch and look-back reads use
// atomicAdd(...,0) so descriptor traffic is device-coherent without extra
// fences. Value width => numElements < 2^30. Each pass has its own contiguous
// region (stateBase); the trailing slot of each region is the tile counter.
#include "ComputeOpsShared.h"

#define BLOCK 256
#define RADIX 256
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
    __shared__ uint sHist[RADIX];       // local digit counts
    __shared__ uint sDigit[BLOCK];      // digit per element (index order)
    __shared__ uint sValid[BLOCK];
    __shared__ uint sExclusive[RADIX];  // look-back exclusive prefix per digit
    __shared__ uint sTile;

    uint tid = threadIdx.x;

    // This pass's look-back region: descriptors + a trailing counter slot.
    uint stateStride = pc.numTiles * RADIX + 1u;
    uint stateBase = pc.passIndex * stateStride;

    // Claim a tile id for forward progress (earlier-scheduled blocks get lower
    // ids and only ever spin on strictly-lower ids).
    if (tid == 0) {
        sTile = atomicAdd(&lookbackState[stateBase + pc.numTiles * RADIX], 1u);
    }
    __syncthreads();
    uint tile = sTile;

    uint bitOffset = pc.passIndex * 8u;
    uint idx = tile * BLOCK + tid;

    sHist[tid] = 0u;
    __syncthreads();

    uint key = 0u, val = 0u, digit = 0u, valid = 0u;
    if (idx < pc.numElements) {
        key = keysIn[idx];
        val = valsIn[idx];
        digit = (key >> bitOffset) & 0xFFu;
        valid = 1u;
        atomicAdd(&sHist[digit], 1u);
    }
    sDigit[tid] = digit;
    sValid[tid] = valid;
    __syncthreads();

    // Thread `tid` handles digit d == tid: publish aggregate, look back, publish
    // inclusive prefix.
    uint d = tid;
    uint agg = sHist[d];
    if (tile == 0u) {
        // No predecessors: exclusive prefix is 0, aggregate is inclusive.
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

    if (valid == 1u) {
        // Stable within-tile rank: earlier same-digit elements in index order.
        uint localRank = 0u;
        for (uint j = 0; j < tid; j++) {
            if (sValid[j] == 1u && sDigit[j] == digit) {
                localRank++;
            }
        }
        uint pos = globalHist[pc.passIndex * RADIX + digit] + sExclusive[digit] +
                   localRank;
        keysOut[pos] = key;
        valsOut[pos] = val;
    }
}
