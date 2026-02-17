#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numIndices;
    uint embDim;
};

// Indices buffer (always uint/int)
layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIndices {
    uint indices[];
};

// Weight table: [num_embeddings, embDim]
layout(set = 0, binding = 1, std430) restrict readonly buffer BufferWeight {
    %VEC_DTYPE% weight_data[];
};

// Output: [numIndices, embDim]
layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% output_data[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;

    uint alignedDim4 = ((embDim + 3) & ~3u) / 4;  // vec4 elements per row
    uint totalVecElements = numIndices * alignedDim4;
    if (gid >= totalVecElements) return;

    uint idx = gid / alignedDim4;    // which index
    uint dim4 = gid % alignedDim4;   // which vec4 chunk

    uint embIdx = indices[idx];

    // Copy vec4 from weight table to output
    output_data[idx * alignedDim4 + dim4] = weight_data[embIdx * alignedDim4 + dim4];
}
