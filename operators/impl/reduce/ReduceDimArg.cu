// Native CUDA counterpart of ReduceDimArg.shader (per-output serial dim
// argmax/argmin). Hash-aliased across dtypes: pure float, no CUT_DTYPE_*.
#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (OP_REDUCE_ARGMAX)
#endif
static const uint op_enum = CUT_SPEC_1;

struct PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

__device__ bool isBetter(float candidate, float current) {
    return op_enum == OP_REDUCE_ARGMAX ? candidate > current : candidate < current;
}

__device__ float worstVal() {
    return op_enum == OP_REDUCE_ARGMAX ? -3.402823466e+38f : 3.402823466e+38f;
}

extern "C" __global__ void cut_main(const float* __restrict__ dataIn, float* __restrict__ dataOut, PushConstants pc) {
    uint outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint numOutputs = pc.outerSize * pc.innerSize;
    if (outIdx >= numOutputs) return;

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    float bestVal = worstVal();
    uint bestIdx = 0;
    for (uint r = 0; r < pc.reduceSize; r++) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        float b = dataIn[inIdx];
        if (isBetter(b, bestVal)) {
            bestVal = b;
            bestIdx = r;
        }
    }
    dataOut[outIdx] = (float)bestIdx;
}
