/*
 * Shared machinery for the native binary-op kernel family.
 *
 * Per-variant dtype macros (CUT_VEC_DTYPE_<SLOT>, CUT_DTYPE_<SLOT>_IS_*) are
 * passed by NVRTC -D defines from the native manifest. Every macro has an
 * #ifndef default so the file reads standalone.
 */

#pragma once

#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

#ifndef CUT_VEC_DTYPE_OUTPUT
#define CUT_VEC_DTYPE_OUTPUT float4
#ifndef CUT_DTYPE_OUTPUT_IS_FLOAT
#define CUT_DTYPE_OUTPUT_IS_FLOAT 1
#endif
#endif

typedef CUT_VEC_DTYPE_OUTPUT cut_out_vec;

#if CUT_DTYPE_OUTPUT_IS_HALF
#define CUT_BIN_OUT_CAST cut_cast_h4
#define CUT_BIN_OUT_IS_FLOAT 1
#define CUT_BIN_OUT_IS_HALF 1
#elif CUT_DTYPE_OUTPUT_IS_FLOAT
#define CUT_BIN_OUT_CAST cut_cast_f4
#define CUT_BIN_OUT_IS_FLOAT 1
#define CUT_BIN_OUT_IS_HALF 0
#elif CUT_DTYPE_OUTPUT_IS_UINT
#define CUT_BIN_OUT_CAST cut_cast_u4
#define CUT_BIN_OUT_IS_FLOAT 0
#define CUT_BIN_OUT_IS_HALF 0
#else /* INT or INT8 (both use int4) */
#define CUT_BIN_OUT_CAST cut_cast_i4
#define CUT_BIN_OUT_IS_FLOAT 0
#define CUT_BIN_OUT_IS_HALF 0
#endif

#ifndef CUT_VEC_DTYPE_INPUT
#define CUT_VEC_DTYPE_INPUT CUT_VEC_DTYPE_OUTPUT
#endif
typedef CUT_VEC_DTYPE_INPUT cut_in_vec;
#if CUT_DTYPE_INPUT_IS_HALF
#define CUT_BIN_IN_CAST cut_cast_h4
#elif CUT_DTYPE_INPUT_IS_FLOAT
#define CUT_BIN_IN_CAST cut_cast_f4
#elif CUT_DTYPE_INPUT_IS_UINT
#define CUT_BIN_IN_CAST cut_cast_u4
#elif CUT_DTYPE_INPUT_IS_INT || CUT_DTYPE_INPUT_IS_INT8
#define CUT_BIN_IN_CAST cut_cast_i4
#else
#define CUT_BIN_IN_CAST CUT_BIN_OUT_CAST
#endif

#ifndef CUT_VEC_DTYPE_INPUT1
#define CUT_VEC_DTYPE_INPUT1 CUT_VEC_DTYPE_OUTPUT
#endif
typedef CUT_VEC_DTYPE_INPUT1 cut_in1_vec;
#if CUT_DTYPE_INPUT1_IS_HALF
#define CUT_BIN_IN1_CAST cut_cast_h4
#elif CUT_DTYPE_INPUT1_IS_FLOAT
#define CUT_BIN_IN1_CAST cut_cast_f4
#elif CUT_DTYPE_INPUT1_IS_UINT
#define CUT_BIN_IN1_CAST cut_cast_u4
#elif CUT_DTYPE_INPUT1_IS_INT || CUT_DTYPE_INPUT1_IS_INT8
#define CUT_BIN_IN1_CAST cut_cast_i4
#else
#define CUT_BIN_IN1_CAST CUT_BIN_OUT_CAST
#endif

#ifndef CUT_VEC_DTYPE_INPUT2
#define CUT_VEC_DTYPE_INPUT2 CUT_VEC_DTYPE_OUTPUT
#endif
typedef CUT_VEC_DTYPE_INPUT2 cut_in2_vec;

static __device__ __forceinline__ float cut_bin_widen(half v) { return __half2float(v); }
static __device__ __forceinline__ float cut_bin_widen(float v) { return v; }
static __device__ __forceinline__ int cut_bin_widen(int v) { return v; }
static __device__ __forceinline__ unsigned int cut_bin_widen(unsigned int v) { return v; }

static __device__ __forceinline__ cut_out_vec cut_binary_apply(uint op, cut_out_vec a, cut_out_vec b) {
    switch (op) {
        case OP_BINARY_ADD:
            return a + b;
        case OP_BINARY_SUB:
            return a - b;
        case OP_BINARY_MUL:
            return a * b;
        case OP_BINARY_DIV:
            return a / b;
        case OP_BINARY_MOD:
#if CUT_BIN_OUT_IS_FLOAT
            return (a - b * floor(a / b));
#else
            return a % b;
#endif
        case OP_BINARY_POW:
#if CUT_BIN_OUT_IS_FLOAT
            return pow(a, b);
#else
            return CUT_BIN_OUT_CAST(0);
#endif
        case OP_BINARY_FLOOR_DIV:
#if CUT_BIN_OUT_IS_FLOAT
            return floor(a / b);
#else
            return a / b;
#endif
        case OP_BINARY_EQUAL:
            return CUT_BIN_OUT_CAST(a == b);
        case OP_BINARY_NOT_EQUAL:
            return CUT_BIN_OUT_CAST(a != b);
        case OP_BINARY_LESS:
            return CUT_BIN_OUT_CAST(a < b);
        case OP_BINARY_LESS_EQUAL:
            return CUT_BIN_OUT_CAST(a <= b);
        case OP_BINARY_GREATER:
            return CUT_BIN_OUT_CAST(a > b);
        case OP_BINARY_GREATER_EQUAL:
            return CUT_BIN_OUT_CAST(a >= b);
        case OP_BINARY_MIN:
            return min(a, b);
        case OP_BINARY_MAX:
            return max(a, b);
        case OP_BINARY_BITWISE_AND:
#if CUT_BIN_OUT_IS_HALF
            return asfloat16(asint16(a) & asint16(b));
#elif CUT_BIN_OUT_IS_FLOAT
            return asfloat(asint(a) & asint(b));
#else
            return a & b;
#endif
        case OP_BINARY_BITWISE_OR:
#if CUT_BIN_OUT_IS_HALF
            return asfloat16(asint16(a) | asint16(b));
#elif CUT_BIN_OUT_IS_FLOAT
            return asfloat(asint(a) | asint(b));
#else
            return a | b;
#endif
        case OP_BINARY_BITWISE_XOR:
#if CUT_BIN_OUT_IS_HALF
            return asfloat16(asint16(a) ^ asint16(b));
#elif CUT_BIN_OUT_IS_FLOAT
            return asfloat(asint(a) ^ asint(b));
#else
            return a ^ b;
#endif
        case OP_BINARY_LEFT_SHIFT:
#if CUT_BIN_OUT_IS_HALF
            return asfloat16(asint16(a) << asint16(b));
#elif CUT_BIN_OUT_IS_FLOAT
            return asfloat(asint(a) << asint(b));
#else
            return a << b;
#endif
        case OP_BINARY_RIGHT_SHIFT:
#if CUT_BIN_OUT_IS_HALF
            return asfloat16(asint16(a) >> asint16(b));
#elif CUT_BIN_OUT_IS_FLOAT
            return asfloat(asint(a) >> asint(b));
#else
            return a >> b;
#endif
        case OP_BINARY_LOGICAL_AND:
            return CUT_BIN_OUT_CAST(a != CUT_BIN_OUT_CAST(0)) * CUT_BIN_OUT_CAST(b != CUT_BIN_OUT_CAST(0));
        case OP_BINARY_LOGICAL_OR:
            return min(CUT_BIN_OUT_CAST(a != CUT_BIN_OUT_CAST(0)) + CUT_BIN_OUT_CAST(b != CUT_BIN_OUT_CAST(0)), CUT_BIN_OUT_CAST(1));
        case OP_BINARY_LOGICAL_XOR:
            return CUT_BIN_OUT_CAST((a != CUT_BIN_OUT_CAST(0)) != (b != CUT_BIN_OUT_CAST(0)));
        case OP_BINARY_ATAN2:
#if CUT_BIN_OUT_IS_HALF
            return cut_cast_h4(atan2(cut_cast_f4(a), cut_cast_f4(b)));
#elif CUT_BIN_OUT_IS_FLOAT
            return atan2(a, b);
#else
            return CUT_BIN_OUT_CAST(0);
#endif
        case OP_BINARY_HYPOT:
#if CUT_BIN_OUT_IS_FLOAT
            return sqrt(a * a + b * b);
#else
            return CUT_BIN_OUT_CAST(0);
#endif
        case OP_BINARY_COPYSIGN:
#if CUT_BIN_OUT_IS_FLOAT
            return CUT_BIN_OUT_CAST(sign(b)) * abs(a);
#else
            return CUT_BIN_OUT_CAST(0);
#endif
        case OP_BINARY_FMOD:
#if CUT_BIN_OUT_IS_FLOAT
            return (a - b * floor(a / b));
#else
            return a % b;
#endif
        case OP_BINARY_LEAKY_RELU:
#if CUT_BIN_OUT_IS_FLOAT
            return lerp(b * a, a, CUT_BIN_OUT_CAST(a > CUT_BIN_OUT_CAST(0.0)));
#else
            return CUT_BIN_OUT_CAST(0);
#endif
        case OP_BINARY_PRELU:
#if CUT_BIN_OUT_IS_FLOAT
            return lerp(b * a, a, CUT_BIN_OUT_CAST(a >= CUT_BIN_OUT_CAST(0.0)));
#else
            return CUT_BIN_OUT_CAST(0);
#endif
        case OP_BINARY_HARDSHRINK:
#if CUT_BIN_OUT_IS_FLOAT
            return lerp(CUT_BIN_OUT_CAST(0.0), a, CUT_BIN_OUT_CAST(abs(a) > b));
#else
            return CUT_BIN_OUT_CAST(0);
#endif
        case OP_BINARY_SOFTSHRINK:
#if CUT_BIN_OUT_IS_FLOAT
            return CUT_BIN_OUT_CAST(sign(a)) * max(abs(a) - b, CUT_BIN_OUT_CAST(0.0));
#else
            return CUT_BIN_OUT_CAST(0);
#endif
        case OP_BINARY_LOGADDEXP:
#if CUT_BIN_OUT_IS_FLOAT
            return max(a, b) + log(1.0 + exp(-abs(a - b)));
#else
            return CUT_BIN_OUT_CAST(0);
#endif
        case OP_BINARY_LOGADDEXP2:
#if CUT_BIN_OUT_IS_FLOAT
            return max(a, b) + log2(1.0 + exp2(-abs(a - b)));
#else
            return CUT_BIN_OUT_CAST(0);
#endif
        default:
            return CUT_BIN_OUT_CAST(0);
    }
}

static __device__ __forceinline__ cut_out_vec cut_binary_cmp_apply(uint op, cut_in_vec a, cut_in_vec b) {
    switch (op) {
        case OP_BINARY_EQUAL:
            return CUT_BIN_OUT_CAST(a == b);
        case OP_BINARY_NOT_EQUAL:
            return CUT_BIN_OUT_CAST(a != b);
        case OP_BINARY_LESS:
            return CUT_BIN_OUT_CAST(a < b);
        case OP_BINARY_LESS_EQUAL:
            return CUT_BIN_OUT_CAST(a <= b);
        case OP_BINARY_GREATER:
            return CUT_BIN_OUT_CAST(a > b);
        case OP_BINARY_GREATER_EQUAL:
            return CUT_BIN_OUT_CAST(a >= b);
        default:
            return CUT_BIN_OUT_CAST(0);
    }
}

#if CUT_BIN_OUT_IS_FLOAT
static __device__ __forceinline__ cut_out_vec cut_unary_apply(uint op, cut_out_vec a) {
    switch (op) {
        case OP_UNARY_NEG:
            return -a;
        case OP_UNARY_ABS:
            return abs(a);
        case OP_UNARY_SQRT:
            return sqrt(a);
        case OP_UNARY_SQUARE:
            return a * a;
        case OP_UNARY_RECIPROCAL:
            return 1.0 / a;
        case OP_UNARY_SIGN:
            return CUT_BIN_OUT_CAST(sign(a));
        case OP_UNARY_EXP:
            return exp(a);
        case OP_UNARY_EXP2:
            return exp2(a);
        case OP_UNARY_EXPM1:
            return exp(a) - 1.0;
        case OP_UNARY_LOG:
            return log(a);
        case OP_UNARY_LOG2:
            return log2(a);
        case OP_UNARY_LOG10:
            return log(a) * 0.4342944819;
        case OP_UNARY_LOG1P:
            return log(1.0 + a);
        case OP_UNARY_SIN:
            return sin(a);
        case OP_UNARY_COS:
            return cos(a);
        case OP_UNARY_TAN:
            return tan(a);
        case OP_UNARY_ASIN:
            return asin(a);
        case OP_UNARY_ACOS:
            return acos(a);
        case OP_UNARY_ATAN:
            return atan(a);
        case OP_UNARY_SINH:
            return sinh(a);
        case OP_UNARY_COSH:
            return cosh(a);
        case OP_UNARY_TANH:
            return tanh(a);
        case OP_UNARY_FLOOR:
            return floor(a);
        case OP_UNARY_CEIL:
            return ceil(a);
        case OP_UNARY_ROUND:
            return CUT_BIN_OUT_CAST(sign(a)) * floor(abs(a) + 0.5);
        case OP_UNARY_CBRT:
            return CUT_BIN_OUT_CAST(sign(a)) * pow(abs(a), CUT_BIN_OUT_CAST(0.333333333333333));
        case OP_UNARY_DEGREES:
            return degrees(a);
        case OP_UNARY_RADIANS:
            return radians(a);
        case OP_UNARY_LOGICAL_NOT:
            return CUT_BIN_OUT_CAST(a == CUT_BIN_OUT_CAST(0.0));
        case OP_UNARY_BITWISE_NOT:
#if CUT_BIN_OUT_IS_HALF
            return CUT_BIN_OUT_CAST(0);
#else
            return asfloat(~asint(a));
#endif
        case OP_UNARY_RELU:
            return max(a, 0.0);
        case OP_UNARY_SIGMOID:
            return 1.0 / (1.0 + exp(-a));
        case OP_UNARY_GELU:
            return 0.5 * a * (1.0 + tanh(0.797884560802865 * (a + 0.044715 * a * a * a)));
        case OP_UNARY_SILU:
            return a / (1.0 + exp(-a));
        case OP_UNARY_SOFTPLUS:
            return log(1.0 + exp(a));
        case OP_UNARY_ISNAN:
            return CUT_BIN_OUT_CAST(isnan(a));
        case OP_UNARY_ISINF:
            return CUT_BIN_OUT_CAST(isinf(a));
        case OP_UNARY_RELU6:
            return clamp(a, 0.0, 6.0);
        case OP_UNARY_ELU:
            return lerp(exp(a) - 1.0, a, CUT_BIN_OUT_CAST(a >= CUT_BIN_OUT_CAST(0.0)));
        case OP_UNARY_SELU:
            return 1.0507009873554804934193349852946 * lerp(1.6732632423543772848170429916717 * (exp(a) - 1.0), a, CUT_BIN_OUT_CAST(a >= CUT_BIN_OUT_CAST(0.0)));
        case OP_UNARY_CELU:
            return max(a, 0.0) + min(exp(a) - 1.0, 0.0);
        case OP_UNARY_MISH:
            return a * tanh(log(1.0 + exp(a)));
        case OP_UNARY_HARDSWISH:
            return a * clamp(a + 3.0, 0.0, 6.0) / 6.0;
        case OP_UNARY_HARDSIGMOID:
            return clamp(a / 6.0 + 0.5, 0.0, 1.0);
        case OP_UNARY_HARDTANH:
            return clamp(a, -1.0, 1.0);
        case OP_UNARY_SOFTSIGN:
            return a / (1.0 + abs(a));
        case OP_UNARY_LOGSIGMOID:
            return -log(1.0 + exp(-a));
        case OP_UNARY_TANHSHRINK:
            return a - tanh(a);
        case OP_UNARY_RSQRT:
            return rsqrt(a);
        case OP_UNARY_TRUNC:
            return trunc(a);
        case OP_UNARY_FRAC:
            return frac(a);
        case OP_UNARY_ASINH:
            return log(a + sqrt(a * a + 1.0));
        case OP_UNARY_ACOSH:
            return log(a + sqrt(a * a - 1.0));
        case OP_UNARY_ATANH:
            return 0.5 * log((1.0 + a) / (1.0 - a));
        case OP_UNARY_ISFINITE:
            return CUT_BIN_OUT_CAST(!isnan(a)) * CUT_BIN_OUT_CAST(!isinf(a));
        default:
            return CUT_BIN_OUT_CAST(0);
    }
}
#endif
