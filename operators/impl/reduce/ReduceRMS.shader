#include "ComputeOpsShared.h"

// Single-pass root-mean-square reduction
// Computes RMS = sqrt(mean(x^2)) = sqrt(sum(x^2) / n)

#define WORKGROUP_SIZE 256

struct PushConstants {
    uint numElements;
    uint actualInner;
    uint alignedInner;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

groupshared float sharedData[WORKGROUP_SIZE];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID) {
    uint tid = GTid.x;

    // Phase 1: Each thread accumulates sum of squares via strided loop
    float localSum = 0.0;
    for (uint i = tid; i < pc.numElements; i += WORKGROUP_SIZE) {
        uint row = i / pc.actualInner;
        uint col = i % pc.actualInner;
        uint idx = row * pc.alignedInner + col;
        float val = dataIn[idx];
        localSum += val * val;
    }
    sharedData[tid] = localSum;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Parallel tree reduction in shared memory
    for (uint stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Phase 3: Write RMS = sqrt(sum / n)
    if (tid == 0) {
        dataOut[0] = sqrt(sharedData[0] / float(pc.numElements));
    }
}
