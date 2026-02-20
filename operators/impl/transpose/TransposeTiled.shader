#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Shared memory tiled transpose: TILE_SIZE=%TILE_SIZE%
// Both reads and writes are coalesced; +1 padding avoids bank conflicts.

#define TILE_SIZE %TILE_SIZE%

struct PushConstants {
    uint M;         // logical rows of input
    uint N;         // logical cols of input
    uint strideIn;  // aligned stride for input rows (aligned N)
    uint strideOut; // aligned stride for output rows (aligned M)
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataOut;

groupshared %SCALAR_DTYPE% tile[TILE_SIZE][TILE_SIZE + 1];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    // Coalesced read: each thread reads one element row-major from input
    uint inRow = Gid.y * TILE_SIZE + GTid.y;
    uint inCol = Gid.x * TILE_SIZE + GTid.x;

    if (inRow < pc.M && inCol < pc.N) {
        tile[GTid.y][GTid.x] = dataIn[inRow * pc.strideIn + inCol];
    } else {
        tile[GTid.y][GTid.x] = (%SCALAR_DTYPE%)(0);
    }

    GroupMemoryBarrierWithGroupSync();

    // Coalesced write: transposed tile, swapped group indices
    uint outRow = Gid.x * TILE_SIZE + GTid.y;
    uint outCol = Gid.y * TILE_SIZE + GTid.x;

    if (outRow < pc.N && outCol < pc.M) {
        dataOut[outRow * pc.strideOut + outCol] = tile[GTid.x][GTid.y];
    }
}
