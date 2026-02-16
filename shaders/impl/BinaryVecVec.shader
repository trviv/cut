#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Specialization constants
layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;
layout(constant_id = 1) const uint op_enum = OP_BINARY_VEC_VEC_ADD;

// Push constants
layout(push_constant) uniform PushConstants {
    uint numElements;
} pushConstants;

// Storage buffers
layout(set = 0, binding = 0) readonly buffer DataA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1) readonly buffer DataB {
    %VEC_DTYPE% dataB[];
};

layout(set = 0, binding = 2) writeonly buffer DataOut {
    %VEC_DTYPE% dataOut[];
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

%VEC_DTYPE% binaryOp(%VEC_DTYPE% a, %VEC_DTYPE% b) {
    switch (op_enum) {
        case OP_BINARY_VEC_VEC_ADD:
            return a + b;
        case OP_BINARY_VEC_VEC_SUB:
            return a - b;
        case OP_BINARY_VEC_VEC_MUL:
            return a * b;
        case OP_BINARY_VEC_VEC_DIV:
            return a / b;
        case OP_BINARY_VEC_VEC_MOD:
#ifdef DTYPE_IS_FLOAT
            return mod(a, b);
#else
            return a % b;
#endif
        case OP_BINARY_VEC_VEC_POW:
#ifdef DTYPE_IS_FLOAT
            return pow(a, b);
#else
            return %VEC_DTYPE%(0);
#endif
        case OP_BINARY_VEC_VEC_FLOOR_DIV:
#ifdef DTYPE_IS_FLOAT
            return floor(a / b);
#else
            return a / b;
#endif
        case OP_BINARY_VEC_VEC_EQUAL:
            return %VEC_DTYPE%(equal(a, b));
        case OP_BINARY_VEC_VEC_NOT_EQUAL:
            return %VEC_DTYPE%(notEqual(a, b));
        case OP_BINARY_VEC_VEC_LESS:
            return %VEC_DTYPE%(lessThan(a, b));
        case OP_BINARY_VEC_VEC_LESS_EQUAL:
            return %VEC_DTYPE%(lessThanEqual(a, b));
        case OP_BINARY_VEC_VEC_GREATER:
            return %VEC_DTYPE%(greaterThan(a, b));
        case OP_BINARY_VEC_VEC_GREATER_EQUAL:
            return %VEC_DTYPE%(greaterThanEqual(a, b));
        case OP_BINARY_VEC_VEC_MIN:
            return min(a, b);
        case OP_BINARY_VEC_VEC_MAX:
            return max(a, b);
        case OP_BINARY_VEC_VEC_BITWISE_AND:
#ifdef DTYPE_IS_FLOAT
            return intBitsToFloat(floatBitsToInt(a) & floatBitsToInt(b));
#else
            return a & b;
#endif
        case OP_BINARY_VEC_VEC_BITWISE_OR:
#ifdef DTYPE_IS_FLOAT
            return intBitsToFloat(floatBitsToInt(a) | floatBitsToInt(b));
#else
            return a | b;
#endif
        case OP_BINARY_VEC_VEC_BITWISE_XOR:
#ifdef DTYPE_IS_FLOAT
            return intBitsToFloat(floatBitsToInt(a) ^ floatBitsToInt(b));
#else
            return a ^ b;
#endif
        case OP_BINARY_VEC_VEC_LEFT_SHIFT:
#ifdef DTYPE_IS_FLOAT
            return intBitsToFloat(floatBitsToInt(a) << floatBitsToInt(b));
#else
            return a << b;
#endif
        case OP_BINARY_VEC_VEC_RIGHT_SHIFT:
#ifdef DTYPE_IS_FLOAT
            return intBitsToFloat(floatBitsToInt(a) >> floatBitsToInt(b));
#else
            return a >> b;
#endif
        case OP_BINARY_VEC_VEC_LOGICAL_AND:
            return %VEC_DTYPE%(notEqual(a, %VEC_DTYPE%(0))) * %VEC_DTYPE%(notEqual(b, %VEC_DTYPE%(0)));
        case OP_BINARY_VEC_VEC_LOGICAL_OR:
            return min(%VEC_DTYPE%(notEqual(a, %VEC_DTYPE%(0))) + %VEC_DTYPE%(notEqual(b, %VEC_DTYPE%(0))), %VEC_DTYPE%(1));
        case OP_BINARY_VEC_VEC_LOGICAL_XOR:
            return %VEC_DTYPE%(notEqual(notEqual(a, %VEC_DTYPE%(0)), notEqual(b, %VEC_DTYPE%(0))));
        case OP_BINARY_VEC_VEC_ATAN2:
#ifdef DTYPE_IS_FLOAT
            return atan(a, b);
#else
            return %VEC_DTYPE%(0);
#endif
        case OP_BINARY_VEC_VEC_HYPOT:
#ifdef DTYPE_IS_FLOAT
            return sqrt(a * a + b * b);
#else
            return %VEC_DTYPE%(0);
#endif
        case OP_BINARY_VEC_VEC_COPYSIGN:
#ifdef DTYPE_IS_FLOAT
            return sign(b) * abs(a);
#else
            return %VEC_DTYPE%(0);
#endif
        case OP_BINARY_VEC_VEC_FMOD:
#ifdef DTYPE_IS_FLOAT
            return mod(a, b);
#else
            return a % b;
#endif
        case OP_BINARY_VEC_VEC_LOGADDEXP:
#ifdef DTYPE_IS_FLOAT
            return log(exp(a) + exp(b));
#else
            return %VEC_DTYPE%(0);
#endif
        case OP_BINARY_VEC_VEC_LOGADDEXP2:
#ifdef DTYPE_IS_FLOAT
            return log2(exp2(a) + exp2(b));
#else
            return %VEC_DTYPE%(0);
#endif
        default:
            return %VEC_DTYPE%(0);
    }
}

void main() {
    uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= pushConstants.numElements) {
        return;
    }

    dataOut[index] = binaryOp(dataA[index], dataB[index]);
}
