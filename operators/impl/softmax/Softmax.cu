// Native CUDA counterpart of Softmax.shader — keep semantics in lockstep.
// Fused single-pass softmax: one 256-thread block per slice.
#include "ComputeOpsShared.h"
#include "SoftmaxCommon.cuh"

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT* __restrict__ dataIn,
                                    CUT_SCALAR_DTYPE_INPUT* __restrict__ dataOut,
                                    PushConstants pc) {
    uint sliceIdx = blockIdx.x;
    uint tid = threadIdx.x;
    uint numSlices = pc.outerSize * pc.innerSize;
    if (sliceIdx >= numSlices) {
        return;
    }

    uint outer = sliceIdx / pc.innerSize;
    uint inner = sliceIdx % pc.innerSize;

    // Phase 1: per-thread online normalizer over strided elements (identical to shader)
    float local_m = -3.402823466e+38f;  // -FLT_MAX
    float local_d = 0.0f;
    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        float x = (float)(dataIn[inIdx]);
        float new_m = fmaxf(local_m, x);
        local_d = local_d * expf(local_m - new_m) + expf(x - new_m);
        local_m = new_m;
    }

    // Phase 2: block tree reduction with online merge
    float global_max, global_sumexp;
    cut_block_online_reduce(local_m, local_d, global_max, global_sumexp);

    // Phase 3: write softmax output
    float inv_sumexp = 1.0f / global_sumexp;
    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        uint idx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        float x = (float)(dataIn[idx]);
        dataOut[idx] = (CUT_SCALAR_DTYPE_INPUT)(expf(x - global_max) * inv_sumexp);
    }
}
