#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

// Specialization constants
layout(constant_id = 0) const uint dtype_vec_size = 4;
layout(constant_id = 1) const uint op_type = OP_BINARY_VEC_SCALAR_ADD;

// Push constants
layout(push_constant) uniform PushConstants {
    uint numElements;
    float scalar;
} pushConstants;

// Storage buffers
layout(set = 0, binding = 0) readonly buffer DataA {
    vec4 dataA[];
};

layout(set = 0, binding = 1) writeonly buffer DataOut {
    vec4 dataOut[];
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= pushConstants.numElements) {
        return;
    }

    vec4 a = dataA[index];
    vec4 s = vec4(pushConstants.scalar);
    vec4 result;

    switch (op_type) {
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
            result = mod(a, s);
            break;
        case OP_BINARY_VEC_SCALAR_POW:
            result = pow(a, s);
            break;
        case OP_BINARY_VEC_SCALAR_FLOOR_DIV:
            result = floor(a / s);
            break;
        case OP_BINARY_VEC_SCALAR_EQUAL:
            result = vec4(equal(a, s));
            break;
        case OP_BINARY_VEC_SCALAR_NOT_EQUAL:
            result = vec4(notEqual(a, s));
            break;
        case OP_BINARY_VEC_SCALAR_LESS:
            result = vec4(lessThan(a, s));
            break;
        case OP_BINARY_VEC_SCALAR_LESS_EQUAL:
            result = vec4(lessThanEqual(a, s));
            break;
        case OP_BINARY_VEC_SCALAR_GREATER:
            result = vec4(greaterThan(a, s));
            break;
        case OP_BINARY_VEC_SCALAR_GREATER_EQUAL:
            result = vec4(greaterThanEqual(a, s));
            break;
        case OP_BINARY_VEC_SCALAR_MIN:
            result = min(a, s);
            break;
        case OP_BINARY_VEC_SCALAR_MAX:
            result = max(a, s);
            break;
        default:
            result = vec4(0.0, 0.0, 0.0, 0.0);
            break;
    }

    dataOut[index] = result;
}
