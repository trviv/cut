// Shared header for the native Conv2D kernel family.
// Dtype macros come from the native manifest as NVRTC -D defines; #ifndef
// defaults keep the file readable standalone (float).
#pragma once
#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#ifndef CUT_DTYPE_INPUT_IS_FLOAT
#define CUT_DTYPE_INPUT_IS_FLOAT 1
#endif
#endif

typedef CUT_SCALAR_DTYPE_INPUT cut_conv2d_t;

struct PushConstants {
    uint batchSize;  // N
    uint C_in;       // input channels
    uint H_in;       // input height
    uint W_in;       // input width
    uint C_out;      // output channels
    uint kH;         // kernel height
    uint kW;         // kernel width
    uint strideH;    // stride height
    uint strideW;    // stride width
    uint padH;       // padding height
    uint padW;       // padding width
};
