#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256
#define RADIX 256

// Fused per-digit radix (Vulkan): each workgroup owns one tile of WG_SIZE
// elements and builds a full RADIX-bin histogram for the current 8-bit digit,
// writing ALL bins (including zeros) in digit-major layout
// tileHist[digit * numTiles + tile] so a single exclusive scan over the whole
// spine yields each (digit, tile) global base offset.
struct PushConstants {
    uint numElements;
    uint bitOffset;
    uint numTiles;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> keys;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> tileHist;

groupshared uint localHist[RADIX];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;   // also the digit this thread writes back
    uint tile = Gid.x;

    localHist[tid] = 0;
    GroupMemoryBarrierWithGroupSync();

    // One element per thread (ELEMS_PER_TILE == WG_SIZE).
    uint idx = tile * WG_SIZE + tid;
    if (idx < pc.numElements) {
        uint digit = (keys[idx] >> pc.bitOffset) & 0xFFu;
        InterlockedAdd(localHist[digit], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    // Every thread writes exactly one bin (bin == tid), digit-major.
    tileHist[tid * pc.numTiles + tile] = localHist[tid];
}
