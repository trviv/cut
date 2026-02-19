#include "ComputeOpsShared.h"

#define WORKGROUP_SIZE 256

struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

groupshared float sharedData[WORKGROUP_SIZE];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID) {
    uint tid = GTid.x;

    // Each thread accumulates squared values via strided loop
    float localVal = 0.0;
    for (uint i = tid; i < pc.numElements; i += WORKGROUP_SIZE) {
        float val = dataIn[i];
        localVal += val * val;
    }
    sharedData[tid] = localVal;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction in shared memory
    for (uint stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write sqrt of result
    if (tid == 0) {
        dataOut[0] = sqrt(sharedData[0]);
    }
}
