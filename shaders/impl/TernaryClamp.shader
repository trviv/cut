#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = 4;

layout(push_constant) uniform PushConstants {
    uint numElements;
    %SCALAR_DTYPE% minVal;
    %SCALAR_DTYPE% maxVal;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %VEC_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = clamp(dataIn[index], %VEC_DTYPE%(minVal), %VEC_DTYPE%(maxVal));
}
