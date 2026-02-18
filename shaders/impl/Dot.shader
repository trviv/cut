#include "ComputeOpsShared.h"

struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataA;

[[vk::binding(1, 0)]] StructuredBuffer<float> dataB;

[[vk::binding(2, 0)]] RWStructuredBuffer<float> dataOut;

groupshared float sharedData[256];

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;
    uint gid = DTid.x;

    // Load and multiply
    if (gid < pc.numElements) {
        sharedData[tid] = dataA[gid] * dataB[gid];
    } else {
        sharedData[tid] = 0.0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction
    for (uint stride = 256 / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write per-workgroup partial sum
    if (tid == 0) {
        dataOut[Gid.x] = sharedData[0];
    }
}
