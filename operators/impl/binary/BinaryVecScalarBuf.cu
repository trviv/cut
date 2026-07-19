// Native CUDA counterpart of BinaryVecScalarBuf.shader (vector-scalar buffer binary op).
#include "ComputeOpsShared.h"

#ifndef CUT_VEC_DTYPE_INPUT1
#define CUT_VEC_DTYPE_INPUT1 float4
#endif
#ifndef CUT_SCALAR_DTYPE_INPUT2
#define CUT_SCALAR_DTYPE_INPUT2 float
#endif

#include "BinaryCommon.cuh"

#ifndef CUT_SPEC_0
#define CUT_SPEC_0 (4)
#endif
static const uint dtype_vec_size = CUT_SPEC_0;
#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (OP_BINARY_ADD)
#endif
static const uint op_enum = CUT_SPEC_1;

struct PushConstants {
    uint numElements;
};

extern "C" __global__ void cut_main(const cut_in1_vec* __restrict__ dataA,
                                    const CUT_SCALAR_DTYPE_INPUT2* __restrict__ dataB,
                                    cut_out_vec* __restrict__ dataOut,
                                    PushConstants pc) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index * dtype_vec_size >= pc.numElements) return;
    cut_out_vec a = CUT_BIN_OUT_CAST(dataA[index]);
    cut_out_vec s = CUT_BIN_OUT_CAST(CUT_BIN_IN1_CAST(cut_bin_widen(dataB[0])));
    dataOut[index] = cut_binary_apply(op_enum, a, s);
}
