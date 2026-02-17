#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint srcAlignedInner;
    uint srcActualInner;
    uint dstAlignedInner;
    uint dstActualInner;
    uint totalElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %SCALAR_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %SCALAR_DTYPE% dataOut[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;

    if (gid >= totalElements) {
        return;
    }

    uint srcRow = gid / srcActualInner;
    uint srcCol = gid % srcActualInner;
    uint dstRow = gid / dstActualInner;
    uint dstCol = gid % dstActualInner;

    dataOut[dstRow * dstAlignedInner + dstCol] = dataIn[srcRow * srcAlignedInner + srcCol];
}
