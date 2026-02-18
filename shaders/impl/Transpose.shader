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
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint col = gl_GlobalInvocationID.x;
    uint row4 = gl_GlobalInvocationID.y;

    uint strideOut4 = strideOut / 4;
    if (col >= N || row4 >= strideOut4) return;

    uint baseRow = row4 * 4;

    // Read 4 elements from consecutive input rows, same column
    %VEC_DTYPE% result = %VEC_DTYPE%(0);
    if (baseRow < M)     result[0] = dataIn[baseRow * strideIn + col];
    if (baseRow + 1 < M) result[1] = dataIn[(baseRow + 1) * strideIn + col];
    if (baseRow + 2 < M) result[2] = dataIn[(baseRow + 2) * strideIn + col];
    if (baseRow + 3 < M) result[3] = dataIn[(baseRow + 3) * strideIn + col];

    // Transpose: write vec4 to consecutive output positions
    dataOut[col * strideOut4 + row4] = result;
}
