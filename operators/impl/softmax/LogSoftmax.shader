#include "ComputeOpsShared.h"

// Fused single-pass log-softmax along a dimension
// One workgroup (256 threads) per output slice
// Pass 1: Online normalizer to compute (max, sumexp)
// Pass 2: Write (x - max) - log(sumexp)

%DTYPE_DEFINES_INPUT%

#define WG_SIZE %WG_SIZE%

struct PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> dataOut;

groupshared float shared_max[WG_SIZE];
groupshared float shared_sumexp[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint sliceIdx = Gid.x;
    uint tid = GTid.x;
    uint numSlices = pc.outerSize * pc.innerSize;

    if (sliceIdx >= numSlices) {
        return;
    }

    uint outer = sliceIdx / pc.innerSize;
    uint inner = sliceIdx % pc.innerSize;

    // Phase 1: Online normalizer — compute (max, sumexp) along reduce dim
    float local_m = -3.402823466e+38; // -FLT_MAX
    float local_d = 0.0;

    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        float x = float(dataIn[inIdx]);

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

    // Phase 3: Write log-softmax output — all threads read global max/sumexp
    float global_max = shared_max[0];
    float log_sumexp = log(shared_sumexp[0]);

    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        uint idx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        float x = float(dataIn[idx]);
        dataOut[idx] = (%SCALAR_DTYPE_INPUT%)(x - global_max - log_sumexp);
    }
}
