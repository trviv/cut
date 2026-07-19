// Shared header for the native MaxPool2D kernel family.
// Dtype macros come from the native manifest as NVRTC -D defines; #ifndef
// defaults keep the file readable standalone (float). Vector-cast selection
// mirrors the BinaryCommon.cuh pattern (HALF checked before FLOAT because
// Float16 defines both _IS_FLOAT and _IS_HALF).
#pragma once
#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#ifndef CUT_DTYPE_INPUT_IS_FLOAT
#define CUT_DTYPE_INPUT_IS_FLOAT 1
#endif
#endif

#ifndef CUT_VEC_DTYPE_INPUT
#define CUT_VEC_DTYPE_INPUT float4
#endif

typedef CUT_SCALAR_DTYPE_INPUT cut_pool_t;
typedef CUT_VEC_DTYPE_INPUT cut_pool_vec;

#if CUT_DTYPE_INPUT_IS_HALF
#define CUT_POOL_VCAST cut_cast_h4
#elif CUT_DTYPE_INPUT_IS_FLOAT
#define CUT_POOL_VCAST cut_cast_f4
#elif CUT_DTYPE_INPUT_IS_UINT
#define CUT_POOL_VCAST cut_cast_u4
#else
#define CUT_POOL_VCAST cut_cast_i4
#endif

struct PushConstants {
    uint N;
    uint C;
    uint H_in;
    uint W_in;
    uint kernelH;
    uint kernelW;
    uint strideH;
    uint strideW;
    uint padH;
    uint padW;
};
