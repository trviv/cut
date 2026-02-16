#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = 4;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferCond {
    %VEC_DTYPE% dataCond[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferX {
    %VEC_DTYPE% dataX[];
};

layout(set = 0, binding = 2, std430) restrict readonly buffer BufferY {
    %VEC_DTYPE% dataY[];
};

layout(set = 0, binding = 3, std430) restrict writeonly buffer BufferOut {
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index * dtype_vec_size >= numElements) {
        return;
    }

    // Select from X where condition is non-zero, otherwise from Y
    %VEC_DTYPE% cond = dataCond[index];
#ifdef DTYPE_IS_FLOAT
    dataOut[index] = mix(dataY[index], dataX[index], notEqual(cond, %VEC_DTYPE%(0.0)));
#elif defined(DTYPE_IS_UINT)
    dataOut[index] = mix(dataY[index], dataX[index], notEqual(cond, %VEC_DTYPE%(0u)));
#else
    dataOut[index] = mix(dataY[index], dataX[index], notEqual(cond, %VEC_DTYPE%(0)));
#endif
}
