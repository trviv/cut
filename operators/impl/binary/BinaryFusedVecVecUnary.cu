// Native CUDA counterpart of BinaryFusedVecVecUnary.shader (vector-vector-unary fused binary op).
#include "ComputeOpsShared.h"

#ifndef CUT_VEC_DTYPE_INPUT1
#define CUT_VEC_DTYPE_INPUT1 float4
#endif
#ifndef CUT_VEC_DTYPE_INPUT2
#define CUT_VEC_DTYPE_INPUT2 float4
#endif

#include "BinaryCommon.cuh"

#ifndef CUT_SPEC_0
#define CUT_SPEC_0 (4)
#endif
static const uint dtype_vec_size = CUT_SPEC_0;
#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (OP_BINARY_ADD)
#endif
static const uint op_enum1 = CUT_SPEC_1;
#ifndef CUT_SPEC_2
#define CUT_SPEC_2 (OP_UNARY_NEG)
#endif
static const uint op_enum2 = CUT_SPEC_2;

struct PushConstants {
    uint numElements;
};

extern "C" __global__ void cut_main(const cut_in1_vec* __restrict__ dataA,
                                    const cut_in2_vec* __restrict__ dataB,
                                    cut_out_vec* __restrict__ dataOut,
                                    PushConstants pc) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index * dtype_vec_size >= pc.numElements) return;
    cut_out_vec a = CUT_BIN_OUT_CAST(dataA[index]);
    cut_out_vec b = CUT_BIN_OUT_CAST(dataB[index]);
    cut_out_vec intermediate = cut_binary_apply(op_enum1, a, b);
    dataOut[index] = cut_unary_apply(op_enum2, intermediate);
}
