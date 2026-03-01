#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint numIndices;
    uint embDim;
};
[[vk::push_constant]] PushConstants pc;

// Indices buffer (always uint/int)
[[vk::binding(0, 0)]] StructuredBuffer<uint> indices;

// Weight table: [num_embeddings, embDim]
[[vk::binding(1, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT%> weight_data;

// Output: [numIndices, embDim]
[[vk::binding(2, 0)]] RWStructuredBuffer<%VEC_DTYPE_INPUT%> output_data;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint gid = DTid.x;

    uint alignedDim4 = ((pc.embDim + 3) & ~3u) / 4;  // vec4 elements per row
    uint totalVecElements = pc.numIndices * alignedDim4;
    if (gid >= totalVecElements) return;

    uint idx = gid / alignedDim4;    // which index
    uint dim4 = gid % alignedDim4;   // which vec4 chunk

    uint embIdx = indices[idx];

    // Copy vec4 from weight table to output
    output_data[idx * alignedDim4 + dim4] = weight_data[embIdx * alignedDim4 + dim4];
}
