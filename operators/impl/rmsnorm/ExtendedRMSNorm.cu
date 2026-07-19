// Native CUDA counterpart of ExtendedRMSNorm.shader — keep semantics in lockstep.
// Fused residual-add + RMS normalization over one 256-thread block.
#include "ComputeOpsShared.h"
#include "RMSNormCommon.cuh"

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT* __restrict__ residual_base,
                                    const CUT_SCALAR_DTYPE_INPUT* __restrict__ delta,
                                    const CUT_SCALAR_DTYPE_INPUT* __restrict__ weight,
                                    CUT_SCALAR_DTYPE_INPUT* __restrict__ output,
                                    PushConstants pc) {
    uint tid = threadIdx.x;
    __shared__ float sharedResidual[2048];  // Max cached dim; larger dims recompute

    // Phase 1: residual = base + delta, cache it, and accumulate sum of squares
    float localSum = 0.0f;
    for (uint i = tid; i < pc.dim; i += 256u) {
        float res = (float)(residual_base[i]) + (float)(delta[i]);
        if (i < 2048u) sharedResidual[i] = res;
        localSum += res * res;
    }

    // Phase 2+3: block-wide sum (its barriers also publish sharedResidual),
    // then the normalization scale
    float sumSquares = cut_block_sum_broadcast(localSum);
    float scale = rsqrtf(sumSquares / (float)pc.dim + pc.eps);

    // Phase 4: apply normalization and weight
    for (uint i = tid; i < pc.dim; i += 256u) {
        float res = (i < 2048u) ? sharedResidual[i]
                                : (float)(residual_base[i]) + (float)(delta[i]);
        float wVal = (float)(weight[i]);
        output[i] = (CUT_SCALAR_DTYPE_INPUT)(res * scale * wVal);
    }
}
