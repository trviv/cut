// Native CUDA counterpart of Copy.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif

typedef CUT_SCALAR_DTYPE_INPUT scalar_t;

struct PushConstants {
    uint srcAlignedInner;
    uint srcActualInner;
    uint dstAlignedInner;
    uint dstActualInner;
    uint totalElements;
};

extern "C" __global__ void cut_main(const scalar_t* __restrict__ dataIn,
                                    scalar_t* __restrict__ dataOut,
                                    PushConstants pc) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= pc.totalElements) {
        return;
    }

    uint srcRow = gid / pc.srcActualInner;
    uint srcCol = gid % pc.srcActualInner;
    uint dstRow = gid / pc.dstActualInner;
    uint dstCol = gid % pc.dstActualInner;

    dataOut[dstRow * pc.dstAlignedInner + dstCol] = dataIn[srcRow * pc.srcAlignedInner + srcCol];
}
