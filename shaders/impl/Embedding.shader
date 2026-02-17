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
    %SCALAR_DTYPE% weight_data[];
};

// Output: [numIndices, embDim]
layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %SCALAR_DTYPE% output_data[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;

    uint totalElements = numIndices * embDim;
    if (gid >= totalElements) return;

    uint idx = gid / embDim;    // which index
    uint dim = gid % embDim;    // which embedding dimension

    uint embIdx = indices[idx];
    uint alignedDim = (embDim + 3) & ~3u;

    // Read from weight table
    uint w_offset = embIdx * alignedDim + dim;
    // Write to output
    uint o_offset = idx * alignedDim + dim;

    output_data[o_offset] = weight_data[w_offset];
}
