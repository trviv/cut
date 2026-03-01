#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Shared memory tiled transpose: TILE_SIZE=%TILE_SIZE%
// Both reads and writes are coalesced; +1 padding avoids bank conflicts.

#define TILE_SIZE %TILE_SIZE%

#include "TransposeCommon.shaderh"

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> dataOut;

groupshared %SCALAR_DTYPE_INPUT% tile[TILE_SIZE][TILE_SIZE + 1];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    // Coalesced read: each thread reads one element row-major from input
    uint inRow = Gid.y * TILE_SIZE + GTid.y;
    uint inCol = Gid.x * TILE_SIZE + GTid.x;

    if (inRow < pc.M && inCol < pc.N) {
        tile[GTid.y][GTid.x] = dataIn[inRow * pc.strideIn + inCol];
    } else {
        tile[GTid.y][GTid.x] = (%SCALAR_DTYPE_INPUT%)(0);
    }

    GroupMemoryBarrierWithGroupSync();

    // Coalesced write: transposed tile, swapped group indices
    uint outRow = Gid.x * TILE_SIZE + GTid.y;
    uint outCol = Gid.y * TILE_SIZE + GTid.x;

    if (outRow < pc.N && outCol < pc.M) {
        dataOut[outRow * pc.strideOut + outCol] = tile[GTid.x][GTid.y];
    }
}
