#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

#define WORKGROUP_SIZE 256
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

shared float sharedData[WORKGROUP_SIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;

    // Each thread accumulates squared values via strided loop
    float localVal = 0.0;
    for (uint i = tid; i < numElements; i += WORKGROUP_SIZE) {
        float val = dataIn[i];
        localVal += val * val;
    }
    sharedData[tid] = localVal;
    barrier();

    // Parallel reduction in shared memory
    for (uint stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        barrier();
    }

    // Write sqrt of result
    if (tid == 0) {
        dataOut[0] = sqrt(sharedData[0]);
    }
}
