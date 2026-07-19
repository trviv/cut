// Native CUDA counterpart of Embedding.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

#ifndef CUT_VEC_DTYPE_INPUT
#define CUT_VEC_DTYPE_INPUT float4
#endif

typedef CUT_VEC_DTYPE_INPUT vec_t;

struct PushConstants {
    uint numIndices;
    uint embDim;
};

extern "C" __global__ void cut_main(const uint* __restrict__ indices,
                                    const vec_t* __restrict__ weight_data,
                                    vec_t* __restrict__ output_data,
                                    PushConstants pc) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint alignedDim4 = ((pc.embDim + 3) & ~3u) / 4;  // vec4 elements per row
    uint totalVecElements = pc.numIndices * alignedDim4;
    if (gid >= totalVecElements) return;

    uint idx = gid / alignedDim4;    // which index
    uint dim4 = gid % alignedDim4;   // which vec4 chunk

    uint embIdx = indices[idx];

    // Copy vec4 from weight table to output
    output_data[idx * alignedDim4 + dim4] = weight_data[embIdx * alignedDim4 + dim4];
}
