#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint pos;          // Row index to write at
    uint kvDim;        // Elements per row (actual)
    uint alignedKvDim; // Aligned stride for 2D cache
};
[[vk::push_constant]] PushConstants pc;

// New vector to write [kvDim]
[[vk::binding(0, 0)]] StructuredBuffer<float> newData;
// Cache buffer [maxSeqLen, kvDim] (read-write, modified in place)
[[vk::binding(1, 0)]] RWStructuredBuffer<float> cache;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint gid = DTid.x;
    if (gid >= pc.kvDim) return;

    cache[pc.pos * pc.alignedKvDim + gid] = newData[gid];
}
