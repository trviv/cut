#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint paddedSize;
};

layout(set = 0, binding = 0, std430) readonly restrict buffer SrcKeys {
    float srcKeys[];
};

layout(set = 0, binding = 1, std430) readonly restrict buffer SrcVals {
    uint srcVals[];
};

layout(set = 0, binding = 2, std430) writeonly restrict buffer DstKeys {
    float dstKeys[];
};

layout(set = 0, binding = 3, std430) writeonly restrict buffer DstVals {
    uint dstVals[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= paddedSize) return;
    if (idx < numElements) {
        dstKeys[idx] = srcKeys[idx];
        dstVals[idx] = srcVals[idx];
    } else {
        dstKeys[idx] = 3.402823466e+38;
        dstVals[idx] = 0xFFFFFFFFu;
    }
}
