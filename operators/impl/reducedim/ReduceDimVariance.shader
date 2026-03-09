#include "ComputeOpsShared.h"

// Single-pass Welford variance along a dimension
// One workgroup (256 threads) per output element

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

groupshared float shared_mean[WG_SIZE];
groupshared float shared_m2[WG_SIZE];
groupshared float shared_count[WG_SIZE];

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

    // Phase 1: Each thread accumulates Welford state along reduce dimension
    float t_mean = 0.0;
    float t_m2 = 0.0;
    float t_count = 0.0;

    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        float val = dataIn[inIdx];

        t_count += 1.0;
        float delta1 = val - t_mean;
        t_mean += delta1 / t_count;
        float delta2 = val - t_mean;
        t_m2 += delta1 * delta2;
    }

    shared_mean[tid] = t_mean;
    shared_m2[tid] = t_m2;
    shared_count[tid] = t_count;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Tree reduction with Welford merge
    for (uint stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a_mean = shared_mean[tid];
            float a_m2 = shared_m2[tid];
            float a_count = shared_count[tid];
            float b_mean = shared_mean[tid + stride];
            float b_m2 = shared_m2[tid + stride];
            float b_count = shared_count[tid + stride];

            float new_count = a_count + b_count;
            if (new_count > 0.0) {
                float delta = b_mean - a_mean;
                shared_mean[tid] = a_mean + delta * b_count / new_count;
                shared_m2[tid] = a_m2 + b_m2 + delta * delta * a_count * b_count / new_count;
            }
            shared_count[tid] = new_count;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Phase 3: Write variance = M2 / count
    if (tid == 0) {
        float count = shared_count[0];
        dataOut[outIdx] = (count > 0.0) ? (shared_m2[0] / count) : 0.0;
    }
}
