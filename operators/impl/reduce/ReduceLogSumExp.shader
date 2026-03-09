#include "ComputeOpsShared.h"

// Single-pass numerically stable LogSumExp reduction
// Computes log(sum(exp(x_i))) using the online normalizer algorithm
// (Milakov & Gimelshein, 2018).
//
// Maintains running (max, sumexp) state where:
//   sumexp = sum(exp(x_i - max))
// When a new max is found, all accumulated sums are rescaled.
// Final result: max + log(sumexp)

#define WORKGROUP_SIZE 256

struct PushConstants {
    uint numElements;
    uint actualInner;
    uint alignedInner;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

groupshared float shared_max[WORKGROUP_SIZE];
groupshared float shared_sumexp[WORKGROUP_SIZE];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID) {
    uint tid = GTid.x;

    // Phase 1: Each thread accumulates online (max, sumexp) state
    float local_m = -3.402823466e+38; // -FLT_MAX
    float local_d = 0.0;

    for (uint i = tid; i < pc.numElements; i += WORKGROUP_SIZE) {
        uint row = i / pc.actualInner;
        uint col = i % pc.actualInner;
        uint idx = row * pc.alignedInner + col;
        float x = dataIn[idx];

        float new_m = max(local_m, x);
        local_d = local_d * exp(local_m - new_m) + exp(x - new_m);
        local_m = new_m;
    }

    shared_max[tid] = local_m;
    shared_sumexp[tid] = local_d;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Tree reduction with online merge
    for (uint stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
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
        dataOut[0] = shared_max[0] + log(shared_sumexp[0]);
    }
}
