#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256

struct PushConstants {
    uint numElements;
    uint isExclusive;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> partialSums;

groupshared float sharedData[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;
    uint gid = Gid.x;
    uint idx = gid * WG_SIZE + tid;

    // Load to shared memory
    sharedData[tid] = (idx < pc.numElements) ? dataIn[idx] : 0.0;
    GroupMemoryBarrierWithGroupSync();

    // Hillis-Steele inclusive scan
    for (uint offset = 1; offset < WG_SIZE; offset <<= 1) {
        float val = (tid >= offset) ? sharedData[tid - offset] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        sharedData[tid] += val;
        GroupMemoryBarrierWithGroupSync();
    }

    // Write output
    if (idx < pc.numElements) {
        if (pc.isExclusive != 0u) {
            dataOut[idx] = (tid > 0) ? sharedData[tid - 1] : 0.0;
        } else {
            dataOut[idx] = sharedData[tid];
        }
    }

    // Last thread writes workgroup total to partial sums
    if (tid == WG_SIZE - 1) {
        partialSums[gid] = sharedData[WG_SIZE - 1];
    }
}
