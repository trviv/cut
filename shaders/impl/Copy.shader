#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint srcAlignedInner;
    uint dstAlignedInner;
    uint actualInnerDim;
    uint numRows;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %SCALAR_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %SCALAR_DTYPE% dataOut[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint totalElements = numRows * actualInnerDim;

    if (gid >= totalElements) {
        return;
    }

    uint row = gid / actualInnerDim;
    uint col = gid % actualInnerDim;

    dataOut[row * dstAlignedInner + col] = dataIn[row * srcAlignedInner + col];
}
