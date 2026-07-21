#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256
#define RADIX 256

// Fused per-digit radix scatter (Vulkan): each workgroup owns one tile, does a
// LOCAL stable rank of its elements by the current 8-bit digit, then scatters
// to the global position tileHist[digit * numTiles + tile] + localRank. The
// scanned tileHist already holds the global base for (digit, tile), so adding
// the stable within-tile rank gives a globally stable ordering.
struct PushConstants {
    uint numElements;
    uint bitOffset;
    uint numTiles;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> keysIn;
[[vk::binding(1, 0)]] StructuredBuffer<uint> valsIn;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> keysOut;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> valsOut;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> scannedTileHist;

groupshared uint sDigit[WG_SIZE];
groupshared uint sValid[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;
    uint tile = Gid.x;
    uint idx = tile * WG_SIZE + tid;

    uint key = 0u;
    uint val = 0u;
    uint digit = 0u;
    uint valid = 0u;
    if (idx < pc.numElements) {
        key = keysIn[idx];
        val = valsIn[idx];
        digit = (key >> pc.bitOffset) & 0xFFu;
        valid = 1u;
    }
    sDigit[tid] = digit;
    sValid[tid] = valid;
    GroupMemoryBarrierWithGroupSync();

    if (valid == 1u) {
        // Stable within-tile rank: count earlier same-digit valid elements.
        uint localRank = 0u;
        for (uint j = 0u; j < tid; j++) {
            if (sValid[j] == 1u && sDigit[j] == digit) {
                localRank++;
            }
        }
        uint pos = scannedTileHist[digit * pc.numTiles + tile] + localRank;
        keysOut[pos] = key;
        valsOut[pos] = val;
    }
}
