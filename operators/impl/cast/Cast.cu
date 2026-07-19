// Native CUDA counterpart of Cast.shader. Dtype selection via NVRTC -D defines.

#include "ComputeOpsShared.h"

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif
#ifndef CUT_SCALAR_DTYPE_OUTPUT
#define CUT_SCALAR_DTYPE_OUTPUT float
#endif

struct PushConstants {
    uint alignedInner;
    uint actualInner;
    uint totalElements;
};

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT* __restrict__ dataIn, CUT_SCALAR_DTYPE_OUTPUT* __restrict__ dataOut, PushConstants pc) {
    uint3 DTid;
    DTid.x = blockIdx.x * blockDim.x + threadIdx.x; DTid.y = blockIdx.y * blockDim.y + threadIdx.y; DTid.z = blockIdx.z * blockDim.z + threadIdx.z;

    uint gid = DTid.x;
    if (gid >= pc.totalElements) {
        return;
    }

    uint row = gid / pc.actualInner;
    uint col = gid % pc.actualInner;
    uint idx = row * pc.alignedInner + col;

    dataOut[idx] = (CUT_SCALAR_DTYPE_OUTPUT)(dataIn[idx]);
}
