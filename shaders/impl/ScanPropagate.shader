#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

#define WG_SIZE 256
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer PartialSums {
    float partialSums[];
};

layout(set = 0, binding = 1, std430) restrict buffer BufferOut {
    float dataOut[];
};

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;
    uint idx = gid * WG_SIZE + tid;

    if (idx < numElements && gid > 0) {
        dataOut[idx] += partialSums[gid];
    }
}
