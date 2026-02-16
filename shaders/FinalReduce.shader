#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint originalNumElements;
    uint reduceOp;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %SCALAR_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %SCALAR_DTYPE% dataOut[];
};

shared %SCALAR_DTYPE% sharedData[256];

%SCALAR_DTYPE% identity() {
    switch (reduceOp) {
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

%SCALAR_DTYPE% reduce(%SCALAR_DTYPE% a, %SCALAR_DTYPE% b) {
    switch (reduceOp) {
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

    %SCALAR_DTYPE% localVal = identity();
    for (uint i = tid; i < numElements; i += 256) {
        localVal = reduce(localVal, dataIn[i]);
    }
    sharedData[tid] = localVal;
    barrier();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = reduce(sharedData[tid], sharedData[tid + stride]);
        }
        barrier();
    }

    if (tid == 0) {
        %SCALAR_DTYPE% result = sharedData[0];
        if (reduceOp == OP_REDUCE_MEAN) {
            result = result / %SCALAR_DTYPE%(originalNumElements);
        }
        dataOut[0] = result;
    }
}
