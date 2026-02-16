#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

#define WG_SIZE 256
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint isExclusive;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer PartialSums {
    float partialSums[];
};

shared float sharedData[WG_SIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;
    uint idx = gid * WG_SIZE + tid;

    // Load to shared memory
    sharedData[tid] = (idx < numElements) ? dataIn[idx] : 0.0;
    barrier();

    // Hillis-Steele inclusive scan
    for (uint offset = 1; offset < WG_SIZE; offset <<= 1) {
        float val = (tid >= offset) ? sharedData[tid - offset] : 0.0;
        barrier();
        sharedData[tid] += val;
        barrier();
    }

    // Write output
    if (idx < numElements) {
        if (isExclusive != 0u) {
            dataOut[idx] = (tid > 0) ? sharedData[tid - 1] : 0.0;
        } else {
            dataOut[idx] = sharedData[tid];
        }
    }

    // Last thread writes workgroup total to partial sums
    if (tid == WG_SIZE - 1) {
        partialSums[gid] = sharedData[WG_SIZE - 1];
    }
}
