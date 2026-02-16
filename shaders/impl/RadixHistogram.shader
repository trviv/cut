#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

#define WG_SIZE 256
#define RADIX 16
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint bitOffset;
    uint groupCount;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer Keys {
    uint keys[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer Histogram {
    uint histogram[];
};

shared uint localHist[RADIX];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;

    // Clear shared histogram
    if (tid < RADIX) {
        localHist[tid] = 0;
    }
    barrier();

    // Count digits for this workgroup's elements
    for (uint i = gid * WG_SIZE + tid; i < numElements; i += WG_SIZE * groupCount) {
        uint digit = (keys[i] >> bitOffset) & 0xFu;
        atomicAdd(localHist[digit], 1);
    }
    barrier();

    // Write local histogram to global memory
    // Layout: histogram[digit * groupCount + gid]
    if (tid < RADIX) {
        histogram[tid * groupCount + gid] = localHist[tid];
    }
}
