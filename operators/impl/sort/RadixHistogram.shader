#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

#define WG_SIZE 256
#define RADIX 16

struct PushConstants {
    uint numElements;
    uint bitOffset;
    uint groupCount;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> keys;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> histogram;

groupshared uint localHist[RADIX];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;
    uint gid = Gid.x;

    // Clear shared histogram
    if (tid < RADIX) {
        localHist[tid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Count digits for this workgroup's elements
    for (uint i = gid * WG_SIZE + tid; i < pc.numElements; i += WG_SIZE * pc.groupCount) {
        uint digit = (keys[i] >> pc.bitOffset) & 0xFu;
        InterlockedAdd(localHist[digit], 1);
    }
    GroupMemoryBarrierWithGroupSync();

    // Write local histogram to global memory
    // Layout: histogram[digit * groupCount + gid]
    if (tid < RADIX) {
        histogram[tid * pc.groupCount + gid] = localHist[tid];
    }
}
