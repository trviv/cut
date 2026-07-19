// Native CUDA counterpart of Expand.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif

typedef CUT_SCALAR_DTYPE_INPUT scalar_t;

struct PushConstants {
    uint ndim;
    uint inShape[4];
    uint outShape[4];
    uint totalElements;
};

extern "C" __global__ void cut_main(const scalar_t* __restrict__ dataIn,
                                    scalar_t* __restrict__ dataOut,
                                    PushConstants pc) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pc.totalElements) return;

    uint outAlignedInner = (pc.outShape[pc.ndim - 1] + 3) & ~3u;
    uint inAlignedInner = (pc.inShape[pc.ndim - 1] + 3) & ~3u;

    // Decompose gid into N-dim coordinates using logical output shape
    uint coords[4];
    uint remaining = gid;
    for (int d = int(pc.ndim) - 1; d >= 0; d--) {
        coords[d] = remaining % pc.outShape[d];
        remaining = remaining / pc.outShape[d];
    }

    // Compute output buffer offset using aligned innermost stride
    uint outStrides[4];
    outStrides[pc.ndim - 1] = 1;
    if (pc.ndim >= 2) {
        outStrides[pc.ndim - 2] = outAlignedInner;
        for (int d = int(pc.ndim) - 3; d >= 0; d--) {
            outStrides[d] = outStrides[d + 1] * pc.outShape[d + 1];
        }
    }
    uint outIdx = 0;
    for (uint d = 0; d < pc.ndim; d++) {
        outIdx += coords[d] * outStrides[d];
    }

    // Clamp coords for broadcast dims (input size 1 -> coord 0)
    uint inCoords[4];
    for (uint d = 0; d < pc.ndim; d++) {
        inCoords[d] = (pc.inShape[d] == 1) ? 0 : coords[d];
    }

    // Compute input buffer offset using aligned innermost stride
    uint inStrides[4];
    inStrides[pc.ndim - 1] = 1;
    if (pc.ndim >= 2) {
        inStrides[pc.ndim - 2] = inAlignedInner;
        for (int d = int(pc.ndim) - 3; d >= 0; d--) {
            inStrides[d] = inStrides[d + 1] * pc.inShape[d + 1];
        }
    }
    uint inIdx = 0;
    for (uint d = 0; d < pc.ndim; d++) {
        inIdx += inCoords[d] * inStrides[d];
    }

    dataOut[outIdx] = dataIn[inIdx];
}
