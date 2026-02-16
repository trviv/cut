#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = 4;

layout(push_constant) uniform PushConstants {
    %SCALAR_DTYPE% start;
    %SCALAR_DTYPE% step;
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict writeonly buffer BufferOut {
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint baseIdx = index * dtype_vec_size;
    if (baseIdx >= numElements) {
        return;
    }

    // Generate 4 consecutive values
    %VEC_DTYPE% result;
    for (uint i = 0; i < dtype_vec_size && (baseIdx + i) < numElements; i++) {
        result[i] = start + %SCALAR_DTYPE%(baseIdx + i) * step;
    }
    dataOut[index] = result;
}
