// Native CUDA counterpart of RMSNorm.shader — keep semantics in lockstep.
// Dispatch: (256, batchSize, 1). Each Y-block normalizes one row.
#include "ComputeOpsShared.h"
#include "RMSNormCommon.cuh"

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT* __restrict__ x,
                                    const CUT_SCALAR_DTYPE_INPUT* __restrict__ weight,
                                    CUT_SCALAR_DTYPE_INPUT* __restrict__ output,
                                    PushConstants pc) {
    uint tid = threadIdx.x;
    uint row = blockIdx.y;              // Row index (0 for 1D input)
    uint rowOff = row * pc.alignedDim;  // Aligned row offset

    // Phase 1: sum of squares for this row (float accumulation for all dtypes)
    float localSum = 0.0f;
    for (uint i = tid; i < pc.dim; i += 256u) {
        float val = (float)(x[rowOff + i]);
        localSum += val * val;
    }

    // Phase 2+3: block-wide sum, then the normalization scale (every thread
    // computes the identical value from the broadcast total)
    float sumSquares = cut_block_sum_broadcast(localSum);
    float scale = rsqrtf(sumSquares / (float)pc.dim + pc.eps);

    // Phase 4: apply normalization and weight (weight is shared across rows)
    for (uint i = tid; i < pc.dim; i += 256u) {
        float xVal = (float)(x[rowOff + i]);
        float wVal = (float)(weight[i]);
        output[rowOff + i] = (CUT_SCALAR_DTYPE_INPUT)(xVal * scale * wVal);
    }
}
