#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Specialization constants
layout(constant_id = 1) const uint op_enum = OP_REDUCE_SUM;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %SCALAR_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %SCALAR_DTYPE% dataOut[];
};

shared %SCALAR_DTYPE% sharedData[256];

%SCALAR_DTYPE% identity() {
    switch (op_enum) {
        case OP_REDUCE_SUM:
        case OP_REDUCE_MEAN:
        case OP_REDUCE_ANY:
            return %SCALAR_DTYPE%(0);
        case OP_REDUCE_PROD:
        case OP_REDUCE_ALL:
            return %SCALAR_DTYPE%(1);
        case OP_REDUCE_MIN:
#ifdef DTYPE_IS_FLOAT
            return 3.402823466e+38;
#elif defined(DTYPE_IS_UINT)
            return 4294967295u;
#else
            return 2147483647;
#endif
        case OP_REDUCE_MAX:
#ifdef DTYPE_IS_FLOAT
            return -3.402823466e+38;
#elif defined(DTYPE_IS_UINT)
            return 0u;
#else
            return -2147483648;
#endif
        default:
            return %SCALAR_DTYPE%(0);
    }
}

%SCALAR_DTYPE% reduceOp(%SCALAR_DTYPE% a, %SCALAR_DTYPE% b) {
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
#ifdef DTYPE_IS_FLOAT
            return (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
#else
            return (a != 0 || b != 0) ? 1 : 0;
#endif
        case OP_REDUCE_ALL:
#ifdef DTYPE_IS_FLOAT
            return (a != 0.0 && b != 0.0) ? 1.0 : 0.0;
#else
            return (a != 0 && b != 0) ? 1 : 0;
#endif
        default:
            return a + b;
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x;

    // Each thread reduces multiple elements via strided loop
    %SCALAR_DTYPE% localVal = identity();
    for (uint i = tid; i < numElements; i += 256) {
        localVal = reduceOp(localVal, dataIn[i]);
    }
    sharedData[tid] = localVal;
    barrier();

    // Parallel reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = reduceOp(sharedData[tid], sharedData[tid + stride]);
        }
        barrier();
    }

    // Single workgroup: write result directly
    if (tid == 0) {
        %SCALAR_DTYPE% result = sharedData[0];
        // For mean, divide by total element count
        if (op_enum == OP_REDUCE_MEAN) {
            result = result / %SCALAR_DTYPE%(numElements);
        }
        dataOut[0] = result;
    }
}
