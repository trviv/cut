#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Vec4-optimized tiled transpose: 16x16 workgroup, each thread processes 4 elements.
// Reads 4 contiguous elements per thread (coalesced), transposes via shared memory,
// writes 4 elements to consecutive output rows (coalesced).
// Effective tile: 64 columns x 16 rows input -> 16 columns x 64 rows output.

#include "TransposeCommon.shaderh"

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> dataOut;

groupshared %SCALAR_DTYPE_INPUT% tile[64][16 + 1];

[numthreads(16, 16, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    // Read phase: each thread loads 4 contiguous elements from a row
    uint inRow = Gid.y * 16 + GTid.y;
    uint inColBase = Gid.x * 64 + GTid.x * 4;

    [unroll] for (uint i = 0; i < 4; i++) {
        uint inCol = inColBase + i;
        if (inRow < pc.M && inCol < pc.N) {
            tile[GTid.x * 4 + i][GTid.y] = dataIn[inRow * pc.strideIn + inCol];
        } else {
            tile[GTid.x * 4 + i][GTid.y] = (%SCALAR_DTYPE_INPUT%)(0);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Write phase: each thread writes 4 elements to consecutive output rows
    uint outCol = Gid.y * 16 + GTid.x;
    uint outRowBase = Gid.x * 64 + GTid.y * 4;

    [unroll] for (uint i = 0; i < 4; i++) {
        uint outRow = outRowBase + i;
        if (outRow < pc.N && outCol < pc.M) {
            dataOut[outRow * pc.strideOut + outCol] = tile[GTid.y * 4 + i][GTid.x];
        }
    }
}
