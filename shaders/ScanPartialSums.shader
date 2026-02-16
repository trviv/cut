#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numGroups;
};

layout(set = 0, binding = 0, std430) restrict buffer PartialSums {
    float partialSums[];
};

void main() {
    float sum = 0.0;
    for (uint i = 0; i < numGroups; i++) {
        float val = partialSums[i];
        partialSums[i] = sum;
        sum += val;
    }
}
