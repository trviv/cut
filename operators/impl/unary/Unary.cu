// Native CUDA counterpart of Unary.shader (element-wise unary op switch).
// Dtype-generic: one source, compiled per variant with NVRTC -D defines.
#include "ComputeOpsShared.h"

#if !defined(CUT_DTYPE_INPUT_IS_FLOAT) && !defined(CUT_DTYPE_INPUT_IS_INT) && \
    !defined(CUT_DTYPE_INPUT_IS_UINT) && !defined(CUT_DTYPE_INPUT_IS_INT8)
#define CUT_DTYPE_INPUT_IS_FLOAT 1
#endif
#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif
#ifndef CUT_VEC_DTYPE_INPUT
#define CUT_VEC_DTYPE_INPUT float4
#endif
#ifndef CUT_DTYPE_SIZE_INPUT
#define CUT_DTYPE_SIZE_INPUT 4
#endif

#if defined(CUT_DTYPE_INPUT_IS_HALF)
#define CUT_BCAST_INPUT cut_cast_h4
#elif defined(CUT_DTYPE_INPUT_IS_INT) || defined(CUT_DTYPE_INPUT_IS_INT8)
#define CUT_BCAST_INPUT cut_cast_i4
#elif defined(CUT_DTYPE_INPUT_IS_UINT)
#define CUT_BCAST_INPUT cut_cast_u4
#else
#define CUT_BCAST_INPUT cut_cast_f4
#endif

#ifndef CUT_SPEC_0
#define CUT_SPEC_0 (CUT_DTYPE_SIZE_INPUT)
#endif
static const uint dtype_vec_size = CUT_SPEC_0;
#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (OP_UNARY_NEG)
#endif
static const uint op_enum = CUT_SPEC_1;

struct PushConstants {
    uint numElements;
};

static __device__ __forceinline__ CUT_VEC_DTYPE_INPUT unaryOp(CUT_VEC_DTYPE_INPUT a) {
    switch (op_enum) {

// Basic math
        case OP_UNARY_NEG:
#ifdef CUT_DTYPE_INPUT_IS_UINT
            return CUT_BCAST_INPUT(0) - a;
#else
            return -a;
#endif
        case OP_UNARY_ABS:
#ifdef CUT_DTYPE_INPUT_IS_UINT
            return a;
#else
            return abs(a);
#endif
        case OP_UNARY_SQRT:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return sqrt(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_SQUARE:
            return a * a;
        case OP_UNARY_RECIPROCAL:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return 1.0 / a;
#elif defined(CUT_DTYPE_INPUT_IS_INT)
            return sign(a) / max(abs(a), cut_cast_i4(1));
#else
            return cut_cast_u4(1) / max(a, cut_cast_u4(1));
#endif
        case OP_UNARY_SIGN:
#ifdef CUT_DTYPE_INPUT_IS_UINT
            return min(a, cut_cast_u4(1));
#else
            return CUT_BCAST_INPUT(sign(a));
#endif

// Exponential/Logarithmic
        case OP_UNARY_EXP:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return exp(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_EXP2:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return exp2(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_EXPM1:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return exp(a) - 1.0;
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_LOG:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return log(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_LOG2:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return log2(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_LOG10:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return log(a) * 0.4342944819;
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_LOG1P:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return log(1.0 + a);
#else
            return CUT_BCAST_INPUT(0);
#endif

// Trigonometric
        case OP_UNARY_SIN:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return sin(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_COS:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return cos(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_TAN:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return tan(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_ASIN:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return asin(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_ACOS:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return acos(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_ATAN:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return atan(a);
#else
            return CUT_BCAST_INPUT(0);
#endif

// Hyperbolic
        case OP_UNARY_SINH:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return sinh(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_COSH:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return cosh(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_TANH:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return tanh(a);
#else
            return CUT_BCAST_INPUT(0);
#endif

// Rounding
        case OP_UNARY_FLOOR:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return floor(a);
#else
            return a;
#endif
        case OP_UNARY_CEIL:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return ceil(a);
#else
            return a;
#endif
        case OP_UNARY_ROUND:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return CUT_BCAST_INPUT(sign(a)) * floor(abs(a) + 0.5);
#else
            return a;
#endif

// Special math
        case OP_UNARY_CBRT:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return CUT_BCAST_INPUT(sign(a)) * pow(abs(a), CUT_BCAST_INPUT(0.333333333333333));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_DEGREES:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return degrees(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_RADIANS:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return radians(a);
#else
            return CUT_BCAST_INPUT(0);
#endif

// Logical/Bitwise
        case OP_UNARY_LOGICAL_NOT:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return CUT_BCAST_INPUT(a == CUT_BCAST_INPUT(0.0));
#else
            return CUT_BCAST_INPUT(a == CUT_BCAST_INPUT(0));
#endif
        case OP_UNARY_BITWISE_NOT:
#if defined(CUT_DTYPE_INPUT_IS_FLOAT) && !defined(CUT_DTYPE_INPUT_IS_HALF)
            return asfloat(~asint(a));
#elif !defined(CUT_DTYPE_INPUT_IS_FLOAT)
            return ~a;
#else
            return CUT_BCAST_INPUT(0);
#endif

// Activation functions
        case OP_UNARY_RELU:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return max(a, 0.0);
#elif defined(CUT_DTYPE_INPUT_IS_INT)
            return max(a, cut_cast_i4(0));
#else
            return a;
#endif
        case OP_UNARY_SIGMOID:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return 1.0 / (1.0 + exp(-a));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_GELU:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return 0.5 * a * (1.0 + tanh(0.797884560802865 * (a + 0.044715 * a * a * a)));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_SILU:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return a / (1.0 + exp(-a));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_SOFTPLUS:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return log(1.0 + exp(a));
#else
            return CUT_BCAST_INPUT(0);
#endif

// Check operations
        case OP_UNARY_ISNAN:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return CUT_BCAST_INPUT(isnan(a));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_ISINF:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return CUT_BCAST_INPUT(isinf(a));
#else
            return CUT_BCAST_INPUT(0);
#endif

// Extended activation functions
        case OP_UNARY_RELU6:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return clamp(a, 0.0, 6.0);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_ELU:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return lerp(exp(a) - 1.0, a, CUT_BCAST_INPUT(a >= CUT_BCAST_INPUT(0.0)));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_SELU:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return 1.0507009873554804934193349852946 * lerp(1.6732632423543772848170429916717 * (exp(a) - 1.0), a, CUT_BCAST_INPUT(a >= CUT_BCAST_INPUT(0.0)));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_CELU:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return max(a, 0.0) + min(exp(a) - 1.0, 0.0);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_MISH:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return a * tanh(log(1.0 + exp(a)));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_HARDSWISH:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return a * clamp(a + 3.0, 0.0, 6.0) / 6.0;
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_HARDSIGMOID:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return clamp(a / 6.0 + 0.5, 0.0, 1.0);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_HARDTANH:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return clamp(a, -1.0, 1.0);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_SOFTSIGN:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return a / (1.0 + abs(a));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_LOGSIGMOID:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return -log(1.0 + exp(-a));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_TANHSHRINK:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return a - tanh(a);
#else
            return CUT_BCAST_INPUT(0);
#endif

// Extended math operations
        case OP_UNARY_RSQRT:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return rsqrt(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_TRUNC:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return trunc(a);
#else
            return a;
#endif
        case OP_UNARY_FRAC:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return frac(a);
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_ASINH:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return log(a + sqrt(a * a + 1.0));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_ACOSH:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return log(a + sqrt(a * a - 1.0));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_ATANH:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return 0.5 * log((1.0 + a) / (1.0 - a));
#else
            return CUT_BCAST_INPUT(0);
#endif
        case OP_UNARY_ISFINITE:
#ifdef CUT_DTYPE_INPUT_IS_FLOAT
            return CUT_BCAST_INPUT(!isnan(a)) * CUT_BCAST_INPUT(!isinf(a));
#else
            return CUT_BCAST_INPUT(0);
#endif

        default:
            return CUT_BCAST_INPUT(0);
    }
}

extern "C" __global__ void cut_main(const CUT_VEC_DTYPE_INPUT* __restrict__ dataIn,
                                   CUT_VEC_DTYPE_INPUT* __restrict__ dataOut,
                                   PushConstants pc) {
    uint3 DTid;
    DTid.x = blockIdx.x * blockDim.x + threadIdx.x; DTid.y = blockIdx.y * blockDim.y + threadIdx.y; DTid.z = blockIdx.z * blockDim.z + threadIdx.z;

    uint index = DTid.x;

    if (index * dtype_vec_size >= pc.numElements) {
        return;
    }

    dataOut[index] = unaryOp(dataIn[index]);
}
