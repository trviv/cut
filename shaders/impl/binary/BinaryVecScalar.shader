#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Specialization constants
[[vk::constant_id(0)]] const uint dtype_vec_size = %DTYPE_SIZE%;
[[vk::constant_id(1)]] const uint op_enum = OP_BINARY_VEC_SCALAR_ADD;

// Push constants
struct PushConstants {
    uint numElements;
    %SCALAR_DTYPE% scalar;
};
[[vk::push_constant]] PushConstants pc;

// Storage buffers
[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE%> dataA;
[[vk::binding(1, 0)]] RWStructuredBuffer<%VEC_DTYPE%> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;

    if (index * dtype_vec_size >= pc.numElements) {
        return;
    }

    %VEC_DTYPE% a = dataA[index];
    %VEC_DTYPE% s = (%VEC_DTYPE%)(pc.scalar);
    %VEC_DTYPE% result;

    switch (op_enum) {
        case OP_BINARY_VEC_SCALAR_ADD:
            result = a + s;
            break;
        case OP_BINARY_VEC_SCALAR_SUB:
            result = a - s;
            break;
        case OP_BINARY_VEC_SCALAR_MUL:
            result = a * s;
            break;
        case OP_BINARY_VEC_SCALAR_DIV:
            result = a / s;
            break;
        case OP_BINARY_VEC_SCALAR_MOD:
#ifdef DTYPE_IS_FLOAT
            result = (a - s * floor(a / s));
#else
            result = a % s;
#endif
            break;
        case OP_BINARY_VEC_SCALAR_POW:
#ifdef DTYPE_IS_FLOAT
            result = pow(a, s);
#else
            result = (%VEC_DTYPE%)(0);
#endif
            break;
        case OP_BINARY_VEC_SCALAR_FLOOR_DIV:
#ifdef DTYPE_IS_FLOAT
            result = floor(a / s);
#else
            result = a / s;
#endif
            break;
        case OP_BINARY_VEC_SCALAR_EQUAL:
            result = (%VEC_DTYPE%)(a == s);
            break;
        case OP_BINARY_VEC_SCALAR_NOT_EQUAL:
            result = (%VEC_DTYPE%)(a != s);
            break;
        case OP_BINARY_VEC_SCALAR_LESS:
            result = (%VEC_DTYPE%)(a < s);
            break;
        case OP_BINARY_VEC_SCALAR_LESS_EQUAL:
            result = (%VEC_DTYPE%)(a <= s);
            break;
        case OP_BINARY_VEC_SCALAR_GREATER:
            result = (%VEC_DTYPE%)(a > s);
            break;
        case OP_BINARY_VEC_SCALAR_GREATER_EQUAL:
            result = (%VEC_DTYPE%)(a >= s);
            break;
        case OP_BINARY_VEC_SCALAR_MIN:
            result = min(a, s);
            break;
        case OP_BINARY_VEC_SCALAR_MAX:
            result = max(a, s);
            break;
        case OP_BINARY_VEC_SCALAR_BITWISE_AND:
#ifdef DTYPE_IS_FLOAT
            result = asfloat(asint(a) & asint(s));
#else
            result = a & s;
#endif
            break;
        case OP_BINARY_VEC_SCALAR_BITWISE_OR:
#ifdef DTYPE_IS_FLOAT
            result = asfloat(asint(a) | asint(s));
#else
            result = a | s;
#endif
            break;
        case OP_BINARY_VEC_SCALAR_BITWISE_XOR:
#ifdef DTYPE_IS_FLOAT
            result = asfloat(asint(a) ^ asint(s));
#else
            result = a ^ s;
#endif
            break;
        case OP_BINARY_VEC_SCALAR_LEFT_SHIFT:
#ifdef DTYPE_IS_FLOAT
            result = asfloat(asint(a) << asint(s));
#else
            result = a << s;
#endif
            break;
        case OP_BINARY_VEC_SCALAR_RIGHT_SHIFT:
#ifdef DTYPE_IS_FLOAT
            result = asfloat(asint(a) >> asint(s));
#else
            result = a >> s;
#endif
            break;
        case OP_BINARY_VEC_SCALAR_LOGICAL_AND:
            result = (%VEC_DTYPE%)(a != (%VEC_DTYPE%)(0)) * (%VEC_DTYPE%)(s != (%VEC_DTYPE%)(0));
            break;
        case OP_BINARY_VEC_SCALAR_LOGICAL_OR:
            result = min((%VEC_DTYPE%)(a != (%VEC_DTYPE%)(0)) + (%VEC_DTYPE%)(s != (%VEC_DTYPE%)(0)), (%VEC_DTYPE%)(1));
            break;
        case OP_BINARY_VEC_SCALAR_LOGICAL_XOR:
            result = (%VEC_DTYPE%)((a != (%VEC_DTYPE%)(0)) != (s != (%VEC_DTYPE%)(0)));
            break;
        case OP_BINARY_VEC_SCALAR_ATAN2:
#ifdef DTYPE_IS_FLOAT
            result = atan2(a, s);
#else
            result = (%VEC_DTYPE%)(0);
#endif
            break;
        case OP_BINARY_VEC_SCALAR_HYPOT:
#ifdef DTYPE_IS_FLOAT
            result = sqrt(a * a + s * s);
#else
            result = (%VEC_DTYPE%)(0);
#endif
            break;
        case OP_BINARY_VEC_SCALAR_COPYSIGN:
#ifdef DTYPE_IS_FLOAT
            result = sign(s) * abs(a);
#else
            result = (%VEC_DTYPE%)(0);
#endif
            break;
        case OP_BINARY_VEC_SCALAR_FMOD:
#ifdef DTYPE_IS_FLOAT
            result = (a - s * floor(a / s));
#else
            result = a % s;
#endif
            break;
        case OP_BINARY_VEC_SCALAR_LEAKY_RELU:
#ifdef DTYPE_IS_FLOAT
            result = lerp(s * a, a, (%VEC_DTYPE%)(a > (%VEC_DTYPE%)(0.0)));
#else
            result = (%VEC_DTYPE%)(0);
#endif
            break;
        case OP_BINARY_VEC_SCALAR_PRELU:
#ifdef DTYPE_IS_FLOAT
            result = lerp(s * a, a, (%VEC_DTYPE%)(a >= (%VEC_DTYPE%)(0.0)));
#else
            result = (%VEC_DTYPE%)(0);
#endif
            break;
        case OP_BINARY_VEC_SCALAR_HARDSHRINK:
#ifdef DTYPE_IS_FLOAT
            result = lerp((%VEC_DTYPE%)(0.0), a, (%VEC_DTYPE%)(abs(a) > s));
#else
            result = (%VEC_DTYPE%)(0);
#endif
            break;
        case OP_BINARY_VEC_SCALAR_SOFTSHRINK:
#ifdef DTYPE_IS_FLOAT
            result = sign(a) * max(abs(a) - s, (%VEC_DTYPE%)(0.0));
#else
            result = (%VEC_DTYPE%)(0);
#endif
            break;
        case OP_BINARY_VEC_SCALAR_LOGADDEXP:
#ifdef DTYPE_IS_FLOAT
            result = max(a, s) + log(1.0 + exp(-abs(a - s)));
#else
            result = (%VEC_DTYPE%)(0);
#endif
            break;
        case OP_BINARY_VEC_SCALAR_LOGADDEXP2:
#ifdef DTYPE_IS_FLOAT
            result = max(a, s) + log2(1.0 + exp2(-abs(a - s)));
#else
            result = (%VEC_DTYPE%)(0);
#endif
            break;
        default:
            result = (%VEC_DTYPE%)(0);
            break;
    }

    dataOut[index] = result;
}
