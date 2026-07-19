// Native CUDA counterpart of ReduceDimNaive.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"
#include "ReduceDimCommon.cuh"

extern "C" __global__ void cut_main(const cut_reduce_t* __restrict__ dataIn,
                                    cut_reduce_t* __restrict__ dataOut,
                                    PushConstants pc) {
    uint outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint numOutputs = pc.outerSize * pc.innerSize;
    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    cut_reduce_t val = cut_identity();
    for (uint r = 0; r < pc.reduceSize; r++) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        val = cut_reduce_op(val, dataIn[inIdx]);
    }
    dataOut[outIdx] = cut_finalize_reduce(val, pc.reduceSize);
}
