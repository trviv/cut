#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

// Specialization constants
layout(constant_id = 0) const uint dtype_vec_size = 4;
layout(constant_id = 1) const uint op_type = OP_BINARY_VEC_VEC_ADD;

// Push constants
layout(push_constant) uniform PushConstants {
    uint numElements;
} pushConstants;

// Storage buffers
layout(set = 0, binding = 0) readonly buffer DataA {
    vec4 dataA[];
};

layout(set = 0, binding = 1) readonly buffer DataB {
    vec4 dataB[];
};

layout(set = 0, binding = 2) writeonly buffer DataOut {
    vec4 dataOut[];
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= pushConstants.numElements) {
        return;
    }

    vec4 a = dataA[index];
    vec4 b = dataB[index];
    vec4 result;

    switch (op_type) {
        case OP_BINARY_VEC_VEC_ADD:
            result = a + b;
            break;
        case OP_BINARY_VEC_VEC_SUB:
            result = a - b;
            break;
        case OP_BINARY_VEC_VEC_MUL:
            result = a * b;
            break;
        case OP_BINARY_VEC_VEC_DIV:
            result = a / b;
            break;
        case OP_BINARY_VEC_VEC_MOD:
            result = mod(a, b);
            break;
        case OP_BINARY_VEC_VEC_POW:
            result = pow(a, b);
            break;
        case OP_BINARY_VEC_VEC_FLOOR_DIV:
            result = floor(a / b);
            break;
        case OP_BINARY_VEC_VEC_EQUAL:
            result = vec4(equal(a, b));
            break;
        case OP_BINARY_VEC_VEC_NOT_EQUAL:
            result = vec4(notEqual(a, b));
            break;
        case OP_BINARY_VEC_VEC_LESS:
            result = vec4(lessThan(a, b));
            break;
        case OP_BINARY_VEC_VEC_LESS_EQUAL:
            result = vec4(lessThanEqual(a, b));
            break;
        case OP_BINARY_VEC_VEC_GREATER:
            result = vec4(greaterThan(a, b));
            break;
        case OP_BINARY_VEC_VEC_GREATER_EQUAL:
            result = vec4(greaterThanEqual(a, b));
            break;
        case OP_BINARY_VEC_VEC_MIN:
            result = min(a, b);
            break;
        case OP_BINARY_VEC_VEC_MAX:
            result = max(a, b);
            break;
        default:
            result = vec4(0.0, 0.0, 0.0, 0.0);
            break;
    }

    dataOut[index] = result;
}
