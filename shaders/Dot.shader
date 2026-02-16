#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    float dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    float dataB[];
};

layout(set = 0, binding = 2, std430) restrict buffer BufferOut {
    float dataOut[];
};

shared float sharedData[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;

    // Load and multiply
    if (gid < numElements) {
        sharedData[tid] = dataA[gid] * dataB[gid];
    } else {
        sharedData[tid] = 0.0;
    }
    barrier();

    // Parallel reduction
    for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        barrier();
    }

    // Write per-workgroup partial sum
    if (tid == 0) {
        dataOut[gl_WorkGroupID.x] = sharedData[0];
    }
}
