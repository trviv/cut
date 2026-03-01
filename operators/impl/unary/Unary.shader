#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Specialization constants
[[vk::constant_id(0)]] const uint dtype_vec_size = %DTYPE_SIZE_INPUT%;
[[vk::constant_id(1)]] const uint op_enum = OP_UNARY_NEG;

// Push constants
struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

// Storage buffers
[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT%> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<%VEC_DTYPE_INPUT%> dataOut;

%VEC_DTYPE_INPUT% unaryOp(%VEC_DTYPE_INPUT% a) {
    switch (op_enum) {
        // =====================================================================
        // Basic math (60-65)
        // =====================================================================
        case OP_UNARY_NEG:
#ifdef DTYPE_INPUT_IS_UINT
            return (%VEC_DTYPE_INPUT%)(0) - a;
#else
            return -a;
#endif
        case OP_UNARY_ABS:
#ifdef DTYPE_INPUT_IS_UINT
            return a;
#else
            return abs(a);
#endif
        case OP_UNARY_SQRT:
#ifdef DTYPE_INPUT_IS_FLOAT
            return sqrt(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_SQUARE:
            return a * a;
        case OP_UNARY_RECIPROCAL:
#ifdef DTYPE_INPUT_IS_FLOAT
            return 1.0 / a;
#elif defined(DTYPE_INPUT_IS_INT)
            return sign(a) / max(abs(a), (int4)(1));
#else
            return (uint4)(1) / max(a, (uint4)(1));
#endif
        case OP_UNARY_SIGN:
#ifdef DTYPE_INPUT_IS_UINT
            return min(a, (uint4)(1));
#else
            return (%VEC_DTYPE_INPUT%)(sign(a));
#endif

        // =====================================================================
        // Exponential/Logarithmic (66-72)
        // =====================================================================
        case OP_UNARY_EXP:
#ifdef DTYPE_INPUT_IS_FLOAT
            return exp(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_EXP2:
#ifdef DTYPE_INPUT_IS_FLOAT
            return exp2(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_EXPM1:
#ifdef DTYPE_INPUT_IS_FLOAT
            return exp(a) - 1.0;
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_LOG:
#ifdef DTYPE_INPUT_IS_FLOAT
            return log(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_LOG2:
#ifdef DTYPE_INPUT_IS_FLOAT
            return log2(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_LOG10:
#ifdef DTYPE_INPUT_IS_FLOAT
            return log(a) * 0.4342944819;
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_LOG1P:
#ifdef DTYPE_INPUT_IS_FLOAT
            return log(1.0 + a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif

        // =====================================================================
        // Trigonometric (73-78)
        // =====================================================================
        case OP_UNARY_SIN:
#ifdef DTYPE_INPUT_IS_FLOAT
            return sin(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_COS:
#ifdef DTYPE_INPUT_IS_FLOAT
            return cos(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_TAN:
#ifdef DTYPE_INPUT_IS_FLOAT
            return tan(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_ASIN:
#ifdef DTYPE_INPUT_IS_FLOAT
            return asin(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_ACOS:
#ifdef DTYPE_INPUT_IS_FLOAT
            return acos(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_ATAN:
#ifdef DTYPE_INPUT_IS_FLOAT
            return atan(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif

        // =====================================================================
        // Hyperbolic (79-81)
        // =====================================================================
        case OP_UNARY_SINH:
#ifdef DTYPE_INPUT_IS_FLOAT
            return sinh(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_COSH:
#ifdef DTYPE_INPUT_IS_FLOAT
            return cosh(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_TANH:
#ifdef DTYPE_INPUT_IS_FLOAT
            return tanh(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif

        // =====================================================================
        // Rounding (82-84)
        // =====================================================================
        case OP_UNARY_FLOOR:
#ifdef DTYPE_INPUT_IS_FLOAT
            return floor(a);
#else
            return a;
#endif
        case OP_UNARY_CEIL:
#ifdef DTYPE_INPUT_IS_FLOAT
            return ceil(a);
#else
            return a;
#endif
        case OP_UNARY_ROUND:
#ifdef DTYPE_INPUT_IS_FLOAT
            return (%VEC_DTYPE_INPUT%)(sign(a)) * floor(abs(a) + 0.5);
#else
            return a;
#endif

        // =====================================================================
        // Special math (85-87)
        // =====================================================================
        case OP_UNARY_CBRT:
#ifdef DTYPE_INPUT_IS_FLOAT
            return (%VEC_DTYPE_INPUT%)(sign(a)) * pow(abs(a), (%VEC_DTYPE_INPUT%)(0.333333333333333));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_DEGREES:
#ifdef DTYPE_INPUT_IS_FLOAT
            return degrees(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_RADIANS:
#ifdef DTYPE_INPUT_IS_FLOAT
            return radians(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif

        // =====================================================================
        // Logical/Bitwise (88-89)
        // =====================================================================
        case OP_UNARY_LOGICAL_NOT:
#ifdef DTYPE_INPUT_IS_FLOAT
            return (%VEC_DTYPE_INPUT%)(a == (%VEC_DTYPE_INPUT%)(0.0));
#else
            return (%VEC_DTYPE_INPUT%)(a == (%VEC_DTYPE_INPUT%)(0));
#endif
        case OP_UNARY_BITWISE_NOT:
#if defined(DTYPE_INPUT_IS_FLOAT) && !defined(DTYPE_INPUT_IS_HALF)
            return asfloat(~asint(a));
#elif !defined(DTYPE_INPUT_IS_FLOAT)
            return ~a;
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif

        // =====================================================================
        // Activation functions (90-94)
        // =====================================================================
        case OP_UNARY_RELU:
#ifdef DTYPE_INPUT_IS_FLOAT
            return max(a, 0.0);
#elif defined(DTYPE_INPUT_IS_INT)
            return max(a, (int4)(0));
#else
            return a;
#endif
        case OP_UNARY_SIGMOID:
#ifdef DTYPE_INPUT_IS_FLOAT
            return 1.0 / (1.0 + exp(-a));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_GELU:
#ifdef DTYPE_INPUT_IS_FLOAT
            return 0.5 * a * (1.0 + tanh(0.797884560802865 * (a + 0.044715 * a * a * a)));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_SILU:
#ifdef DTYPE_INPUT_IS_FLOAT
            return a / (1.0 + exp(-a));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_SOFTPLUS:
#ifdef DTYPE_INPUT_IS_FLOAT
            return log(1.0 + exp(a));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif

        // =====================================================================
        // Check operations (95-96)
        // =====================================================================
        case OP_UNARY_ISNAN:
#ifdef DTYPE_INPUT_IS_FLOAT
            return (%VEC_DTYPE_INPUT%)(isnan(a));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_ISINF:
#ifdef DTYPE_INPUT_IS_FLOAT
            return (%VEC_DTYPE_INPUT%)(isinf(a));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif

        // =====================================================================
        // Extended activation functions (160-170)
        // =====================================================================
        case OP_UNARY_RELU6:
#ifdef DTYPE_INPUT_IS_FLOAT
            return clamp(a, 0.0, 6.0);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_ELU:
#ifdef DTYPE_INPUT_IS_FLOAT
            return lerp(exp(a) - 1.0, a, (%VEC_DTYPE_INPUT%)(a >= (%VEC_DTYPE_INPUT%)(0.0)));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_SELU:
#ifdef DTYPE_INPUT_IS_FLOAT
            return 1.0507009873554804934193349852946 * lerp(1.6732632423543772848170429916717 * (exp(a) - 1.0), a, (%VEC_DTYPE_INPUT%)(a >= (%VEC_DTYPE_INPUT%)(0.0)));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_CELU:
#ifdef DTYPE_INPUT_IS_FLOAT
            return max(a, 0.0) + min(exp(a) - 1.0, 0.0);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_MISH:
#ifdef DTYPE_INPUT_IS_FLOAT
            return a * tanh(log(1.0 + exp(a)));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_HARDSWISH:
#ifdef DTYPE_INPUT_IS_FLOAT
            return a * clamp(a + 3.0, 0.0, 6.0) / 6.0;
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_HARDSIGMOID:
#ifdef DTYPE_INPUT_IS_FLOAT
            return clamp(a / 6.0 + 0.5, 0.0, 1.0);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_HARDTANH:
#ifdef DTYPE_INPUT_IS_FLOAT
            return clamp(a, -1.0, 1.0);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_SOFTSIGN:
#ifdef DTYPE_INPUT_IS_FLOAT
            return a / (1.0 + abs(a));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_LOGSIGMOID:
#ifdef DTYPE_INPUT_IS_FLOAT
            return -log(1.0 + exp(-a));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_TANHSHRINK:
#ifdef DTYPE_INPUT_IS_FLOAT
            return a - tanh(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif

        // =====================================================================
        // Extended math operations (171-177)
        // =====================================================================
        case OP_UNARY_RSQRT:
#ifdef DTYPE_INPUT_IS_FLOAT
            return rsqrt(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_TRUNC:
#ifdef DTYPE_INPUT_IS_FLOAT
            return trunc(a);
#else
            return a;
#endif
        case OP_UNARY_FRAC:
#ifdef DTYPE_INPUT_IS_FLOAT
            return frac(a);
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_ASINH:
#ifdef DTYPE_INPUT_IS_FLOAT
            return log(a + sqrt(a * a + 1.0));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_ACOSH:
#ifdef DTYPE_INPUT_IS_FLOAT
            return log(a + sqrt(a * a - 1.0));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_ATANH:
#ifdef DTYPE_INPUT_IS_FLOAT
            return 0.5 * log((1.0 + a) / (1.0 - a));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif
        case OP_UNARY_ISFINITE:
#ifdef DTYPE_INPUT_IS_FLOAT
            return (%VEC_DTYPE_INPUT%)(!isnan(a)) * (%VEC_DTYPE_INPUT%)(!isinf(a));
#else
            return (%VEC_DTYPE_INPUT%)(0);
#endif

        default:
            return (%VEC_DTYPE_INPUT%)(0);
    }
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;

    if (index * dtype_vec_size >= pc.numElements) {
        return;
    }

    dataOut[index] = unaryOp(dataIn[index]);
}
