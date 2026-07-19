// Native CUDA counterpart of BinaryVecRowBcast.shader (row-broadcast binary op).
#include "ComputeOpsShared.h"

#ifndef CUT_SCALAR_DTYPE_INPUT1
#define CUT_SCALAR_DTYPE_INPUT1 float
#endif
#ifndef CUT_SCALAR_DTYPE_INPUT2
#define CUT_SCALAR_DTYPE_INPUT2 float
#endif
#ifndef CUT_SCALAR_DTYPE_OUTPUT
#define CUT_SCALAR_DTYPE_OUTPUT float
#endif

#include "BinaryCommon.cuh"

#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (OP_BINARY_ADD)
#endif
static const uint op_enum = CUT_SPEC_1;

struct PushConstants {
    uint numElements;
    uint cols;
    uint alignedCols;
};

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT1* __restrict__ dataA,
                                    const CUT_SCALAR_DTYPE_INPUT2* __restrict__ dataB,
                                    CUT_SCALAR_DTYPE_OUTPUT* __restrict__ dataOut,
                                    PushConstants pc) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pc.numElements) return;
    uint c = i % pc.alignedCols;
    if (c >= pc.cols) return;
    CUT_SCALAR_DTYPE_OUTPUT a = (CUT_SCALAR_DTYPE_OUTPUT)dataA[i];
    CUT_SCALAR_DTYPE_OUTPUT b = (CUT_SCALAR_DTYPE_OUTPUT)dataB[c];
    dataOut[i] = cut_binary_apply(op_enum, CUT_BIN_OUT_CAST(cut_bin_widen(a)),
                                  CUT_BIN_OUT_CAST(cut_bin_widen(b))).x;
}
