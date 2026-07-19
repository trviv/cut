// Native CUDA counterpart of TransposeCommon.shaderh — shared push constants
// and dtype-macro defaults for the transpose kernel family.
#pragma once

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif

#ifndef CUT_VEC_DTYPE_INPUT
#define CUT_VEC_DTYPE_INPUT float4
#endif

#ifndef CUT_DTYPE_INPUT_IS_INT8
#define CUT_DTYPE_INPUT_IS_INT8 0
#endif

struct PushConstants {
    uint M;          // logical rows of input
    uint N;          // logical cols of input
    uint strideIn;   // aligned stride for input rows (aligned N)
    uint strideOut;  // aligned stride for output rows (aligned M)
};
