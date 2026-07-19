// Native CUDA counterpart of Fill.shader. Dtype selection via NVRTC -D defines.

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

#if defined(CUT_DTYPE_INPUT_IS_HALF)
#define CUT_BCAST_INPUT cut_cast_h4
#elif defined(CUT_DTYPE_INPUT_IS_INT) || defined(CUT_DTYPE_INPUT_IS_INT8)
#define CUT_BCAST_INPUT cut_cast_i4
#elif defined(CUT_DTYPE_INPUT_IS_UINT)
#define CUT_BCAST_INPUT cut_cast_u4
#else
#define CUT_BCAST_INPUT cut_cast_f4
#endif

struct PushConstants {
    uint numElements;
    CUT_SCALAR_DTYPE_INPUT fillValue;
};

extern "C" __global__ void cut_main(CUT_VEC_DTYPE_INPUT* __restrict__ dataOut, PushConstants pc) {
    uint3 DTid;
    DTid.x = blockIdx.x * blockDim.x + threadIdx.x; DTid.y = blockIdx.y * blockDim.y + threadIdx.y; DTid.z = blockIdx.z * blockDim.z + threadIdx.z;

    uint index = DTid.x;
    if (index * dtype_vec_size >= pc.numElements) {
        return;
    }

    dataOut[index] = CUT_BCAST_INPUT(pc.fillValue);
}
