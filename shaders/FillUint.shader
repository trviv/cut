#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint fillValue;
};

layout(set = 0, binding = 0, std430) restrict writeonly buffer BufferOut {
    uint dataOut[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < numElements) {
        dataOut[idx] = fillValue;
    }
}
