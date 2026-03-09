#include "ComputeOpsShared.h"

// Single-pass RMS along a dimension
// One workgroup (256 threads) per output element
// Computes sqrt(mean(x^2)) = sqrt(sum(x^2) / reduceSize)

#define WG_SIZE 256

struct PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

groupshared float sharedData[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint outIdx = Gid.x;
    uint tid = GTid.x;
    uint numOutputs = pc.outerSize * pc.innerSize;

    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    // Phase 1: Each thread accumulates sum of squares along reduce dimension
    float localSum = 0.0;
    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        float val = dataIn[inIdx];
        localSum += val * val;
    }
    sharedData[tid] = localSum;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Tree reduction in shared memory
    for (uint stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Phase 3: Write RMS = sqrt(sum / reduceSize)
    if (tid == 0) {
        dataOut[outIdx] = sqrt(sharedData[0] / float(pc.reduceSize));
    }
}
