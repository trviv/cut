#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define BLOCK 256
#define RADIX 256
#define FLAG_AGG (1u << 30)
#define FLAG_INC (2u << 30)
#define FLAG_MASK (3u << 30)
#define VALUE_MASK 0x3FFFFFFFu

// OneSweep decoupled-look-back scatter (Vulkan). One workgroup == one tile of
// BLOCK elements (one element per thread). The tile id is claimed from a global
// atomic counter (not SV_GroupID) so the chained look-back has a forward-
// progress order. Thread `tid` owns digit `tid`: it publishes this tile's
// per-digit aggregate, walks predecessor tiles summing their aggregates until it
// hits an INCLUSIVE descriptor, then publishes this tile's inclusive prefix.
//
// Each element's global position is
//     globalHist[pass*RADIX + digit]  (global base for the digit)
//   + exclusivePrefix[digit]          (same-digit elements in earlier tiles)
//   + localRank                       (stable rank within this tile).
//
// Descriptor packing (lookbackState[tile*RADIX + digit]): value in bits [29:0],
// status flag in bits [31:30] (0=NOT_READY, 1=AGGREGATE, 2=INCLUSIVE). Value
// width => numElements < 2^30. lookbackState[numTiles*RADIX] is the dynamic
// partition/tile counter. Native CUDA counterpart lives in OneSweepScatter.cu;
// semantics kept in lockstep. NOTE: decoupled look-back relies on concurrent
// workgroup forward progress, which Vulkan does not formally guarantee; this is
// validated on the target GPU rather than portable to every driver.
struct PushConstants {
    uint numElements;
    uint passIndex;
    uint numTiles;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> keysIn;
[[vk::binding(1, 0)]] StructuredBuffer<uint> valsIn;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> keysOut;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> valsOut;
[[vk::binding(4, 0)]] StructuredBuffer<uint> globalHist;
[[vk::binding(5, 0)]] globallycoherent RWStructuredBuffer<uint> lookbackState;

groupshared uint sHist[RADIX];
groupshared uint sDigit[BLOCK];
groupshared uint sValid[BLOCK];
groupshared uint sExclusive[RADIX];
groupshared uint sTile;

[numthreads(BLOCK, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID) {
    uint tid = GTid.x;

    // This pass's look-back region: descriptors + a trailing counter slot.
    uint stateStride = pc.numTiles * RADIX + 1u;
    uint stateBase = pc.passIndex * stateStride;

    if (tid == 0) {
        uint claimed;
        InterlockedAdd(lookbackState[stateBase + pc.numTiles * RADIX], 1u, claimed);
        sTile = claimed;
    }
    GroupMemoryBarrierWithGroupSync();
    uint tile = sTile;

    uint bitOffset = pc.passIndex * 8u;
    uint idx = tile * BLOCK + tid;

    sHist[tid] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint key = 0u, val = 0u, digit = 0u, valid = 0u;
    if (idx < pc.numElements) {
        key = keysIn[idx];
        val = valsIn[idx];
        digit = (key >> bitOffset) & 0xFFu;
        valid = 1u;
        InterlockedAdd(sHist[digit], 1u);
    }
    sDigit[tid] = digit;
    sValid[tid] = valid;
    GroupMemoryBarrierWithGroupSync();

    // Thread `tid` handles digit d == tid.
    uint d = tid;
    uint agg = sHist[d];
    uint old;
    if (tile == 0u) {
        InterlockedExchange(lookbackState[stateBase + d],
                            (agg & VALUE_MASK) | FLAG_INC, old);
        sExclusive[d] = 0u;
    } else {
        InterlockedExchange(lookbackState[stateBase + tile * RADIX + d],
                            (agg & VALUE_MASK) | FLAG_AGG, old);
        uint exclusive = 0u;
        int pred = int(tile) - 1;
        [allow_uav_condition]
        while (pred >= 0) {
            uint w = 0u;
            [allow_uav_condition]
            do {
                InterlockedAdd(lookbackState[stateBase + uint(pred) * RADIX + d], 0u, w);
            } while ((w & FLAG_MASK) == 0u);
            exclusive += (w & VALUE_MASK);
            if ((w & FLAG_MASK) == FLAG_INC) {
                break;
            }
            pred--;
        }
        InterlockedExchange(lookbackState[stateBase + tile * RADIX + d],
                            ((exclusive + agg) & VALUE_MASK) | FLAG_INC, old);
        sExclusive[d] = exclusive;
    }
    GroupMemoryBarrierWithGroupSync();

    if (valid == 1u) {
        // Stable within-tile rank: earlier same-digit elements in index order.
        uint localRank = 0u;
        for (uint j = 0u; j < tid; j++) {
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
