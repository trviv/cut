// Native CUDA counterpart of ReduceDimCommon.shaderh — shared push constants,
// dtype defaults, and reduce-op helpers for the reducedim kernel family.
#pragma once

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif
#ifndef CUT_DTYPE_INPUT_IS_FLOAT
#define CUT_DTYPE_INPUT_IS_FLOAT 1
#endif
#ifndef CUT_DTYPE_INPUT_IS_HALF
#define CUT_DTYPE_INPUT_IS_HALF 0
#endif
#ifndef CUT_DTYPE_INPUT_IS_INT
#define CUT_DTYPE_INPUT_IS_INT 0
#endif
#ifndef CUT_DTYPE_INPUT_IS_UINT
#define CUT_DTYPE_INPUT_IS_UINT 0
#endif

#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (OP_REDUCE_SUM)
#endif
static const uint op_enum = CUT_SPEC_1;

struct PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

typedef CUT_SCALAR_DTYPE_INPUT cut_reduce_t;

static __device__ __forceinline__ cut_reduce_t cut_identity() {
    switch (op_enum) {
        case OP_REDUCE_SUM:
        case OP_REDUCE_MEAN:
        case OP_REDUCE_ANY:
        case OP_NORM_DIM:
            return (cut_reduce_t)(0);
        case OP_REDUCE_PROD:
        case OP_REDUCE_ALL:
            return (cut_reduce_t)(1);
        case OP_REDUCE_MIN:
#if CUT_DTYPE_INPUT_IS_HALF
            return __float2half(3.402823466e+38f);  // saturates to +inf: min identity
#elif CUT_DTYPE_INPUT_IS_FLOAT
            return 3.402823466e+38f;
#elif CUT_DTYPE_INPUT_IS_UINT
            return 4294967295u;
#else
            return 2147483647;
#endif
        case OP_REDUCE_MAX:
#if CUT_DTYPE_INPUT_IS_HALF
            return __float2half(-3.402823466e+38f);  // saturates to -inf: max identity
#elif CUT_DTYPE_INPUT_IS_FLOAT
            return -3.402823466e+38f;
#elif CUT_DTYPE_INPUT_IS_UINT
            return 0u;
#else
            return (-2147483647 - 1);
#endif
        default:
            return (cut_reduce_t)(0);
    }
}

static __device__ __forceinline__ cut_reduce_t cut_reduce_op(cut_reduce_t a, cut_reduce_t b) {
    switch (op_enum) {
        case OP_REDUCE_SUM:
        case OP_REDUCE_MEAN:
            return a + b;
        case OP_REDUCE_PROD:
            return a * b;
        case OP_REDUCE_MIN:
            return min(a, b);
        case OP_REDUCE_MAX:
            return max(a, b);
        case OP_REDUCE_ANY:
            // Half branches must compare via __half2float: comparing half against a
            // float/double literal directly is an NVRTC ambiguity error.
#if CUT_DTYPE_INPUT_IS_HALF
            return (__half2float(a) != 0.0f || __half2float(b) != 0.0f) ? (cut_reduce_t)(1) : (cut_reduce_t)(0);
#elif CUT_DTYPE_INPUT_IS_FLOAT
            return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
#else
            return (a != 0 || b != 0) ? 1 : 0;
#endif
        case OP_REDUCE_ALL:
#if CUT_DTYPE_INPUT_IS_HALF
            return (__half2float(a) != 0.0f && __half2float(b) != 0.0f) ? (cut_reduce_t)(1) : (cut_reduce_t)(0);
#elif CUT_DTYPE_INPUT_IS_FLOAT
            return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
#else
            return (a != 0 && b != 0) ? 1 : 0;
#endif
        case OP_NORM_DIM:
            return a + b * b;
        default:
            return a + b;
    }
}

static __device__ __forceinline__ cut_reduce_t cut_finalize_reduce(cut_reduce_t val, uint reduceSize) {
    if (op_enum == OP_REDUCE_MEAN) {
        return val / (cut_reduce_t)(reduceSize);
    } else if (op_enum == OP_NORM_DIM) {
#if CUT_DTYPE_INPUT_IS_HALF
        return hsqrt(val);
#elif CUT_DTYPE_INPUT_IS_FLOAT
        return sqrtf(val);
#endif
    }
    return val;
}
