// Native CUDA counterpart of BinaryVecVecCmp.shader (vector-vector comparison op).
#include "ComputeOpsShared.h"

#ifndef CUT_VEC_DTYPE_INPUT
#define CUT_VEC_DTYPE_INPUT float4
#ifndef CUT_DTYPE_INPUT_IS_FLOAT
#define CUT_DTYPE_INPUT_IS_FLOAT 1
#endif
#endif
#ifndef CUT_VEC_DTYPE_OUTPUT
#define CUT_VEC_DTYPE_OUTPUT uint4
#ifndef CUT_DTYPE_OUTPUT_IS_UINT
#define CUT_DTYPE_OUTPUT_IS_UINT 1
#endif
#endif

#include "BinaryCommon.cuh"

#ifndef CUT_SPEC_0
#define CUT_SPEC_0 (4)
#endif
static const uint dtype_vec_size = CUT_SPEC_0;
#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (OP_BINARY_EQUAL)
#endif
static const uint op_enum = CUT_SPEC_1;

struct PushConstants {
    uint numElements;
};

extern "C" __global__ void cut_main(const cut_in_vec* __restrict__ dataA,
                                    const cut_in_vec* __restrict__ dataB,
                                    cut_out_vec* __restrict__ dataOut,
                                    PushConstants pc) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index * dtype_vec_size >= pc.numElements) return;
    dataOut[index] = cut_binary_cmp_apply(op_enum, dataA[index], dataB[index]);
}
