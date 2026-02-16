#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint M;         // logical rows of input
    uint N;         // logical cols of input
    uint strideIn;  // aligned stride for input rows (aligned N)
    uint strideOut; // aligned stride for output rows (aligned M)
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %SCALAR_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %SCALAR_DTYPE% dataOut[];
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (row < M && col < N) {
        // Transpose: out[col, row] = in[row, col]
        dataOut[col * strideOut + row] = dataIn[row * strideIn + col];
    }
}
