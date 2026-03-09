#include "ComputeOpsShared.h"

// Single-pass LogSumExp along a dimension
// One workgroup (256 threads) per output element
// Uses online normalizer algorithm for numerical stability

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

groupshared float shared_max[WG_SIZE];
groupshared float shared_sumexp[WG_SIZE];

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

    // Phase 1: Each thread accumulates online (max, sumexp) state
    float local_m = -3.402823466e+38; // -FLT_MAX
    float local_d = 0.0;

    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        float x = dataIn[inIdx];

        float new_m = max(local_m, x);
        local_d = local_d * exp(local_m - new_m) + exp(x - new_m);
        local_m = new_m;
    }

    shared_max[tid] = local_m;
    shared_sumexp[tid] = local_d;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Tree reduction with online merge
    for (uint stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a_m = shared_max[tid];
            float a_d = shared_sumexp[tid];
            float b_m = shared_max[tid + stride];
            float b_d = shared_sumexp[tid + stride];

            float new_m = max(a_m, b_m);
            shared_sumexp[tid] = a_d * exp(a_m - new_m) + b_d * exp(b_m - new_m);
            shared_max[tid] = new_m;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Phase 3: Write logsumexp = max + log(sumexp)
    if (tid == 0) {
        dataOut[outIdx] = shared_max[0] + log(shared_sumexp[0]);
    }
}
