// Native CUDA counterpart of Arange.shader. Dtype selection via NVRTC -D defines.

#include "ComputeOpsShared.h"

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif
#ifndef CUT_VEC_DTYPE_INPUT
#define CUT_VEC_DTYPE_INPUT float4
#endif

#ifndef CUT_SPEC_0
#define CUT_SPEC_0 (4)
#endif
static const uint dtype_vec_size = CUT_SPEC_0;

struct PushConstants {
    uint numElements;
    CUT_SCALAR_DTYPE_INPUT start;
    CUT_SCALAR_DTYPE_INPUT step;
};

extern "C" __global__ void cut_main(CUT_VEC_DTYPE_INPUT* __restrict__ dataOut, PushConstants pc) {
    uint3 DTid;
    DTid.x = blockIdx.x * blockDim.x + threadIdx.x; DTid.y = blockIdx.y * blockDim.y + threadIdx.y; DTid.z = blockIdx.z * blockDim.z + threadIdx.z;

    uint index = DTid.x;
    uint baseIdx = index * dtype_vec_size;
    if (baseIdx >= pc.numElements) return;

    CUT_VEC_DTYPE_INPUT result;
    #pragma unroll
    for (uint i = 0; i < dtype_vec_size && (baseIdx + i) < pc.numElements; i++) {
        result[i] = pc.start + (CUT_SCALAR_DTYPE_INPUT)(baseIdx + i) * pc.step;
    }
    dataOut[index] = result;
}
