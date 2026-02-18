#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

#define TILE_SIZE 16

struct PushConstants {
    uint M;  // rows of A
    uint K;  // cols of A / rows of B
    uint N;  // cols of B
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataA;

[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataB;

[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataC;

groupshared %SCALAR_DTYPE% tileA[TILE_SIZE][TILE_SIZE];
groupshared %SCALAR_DTYPE% tileB[TILE_SIZE][TILE_SIZE];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID) {
    uint row = DTid.y;
    uint col = DTid.x;
    uint localRow = GTid.y;
    uint localCol = GTid.x;

    %SCALAR_DTYPE% sum = (%SCALAR_DTYPE%)(0);

    // Loop over tiles
    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        // Load tile from A
        uint aCol = t * TILE_SIZE + localCol;
        if (row < pc.M && aCol < pc.K) {
            tileA[localRow][localCol] = dataA[row * pc.K + aCol];
        } else {
            tileA[localRow][localCol] = (%SCALAR_DTYPE%)(0);
        }

        // Load tile from B
        uint bRow = t * TILE_SIZE + localRow;
        if (bRow < pc.K && col < pc.N) {
            tileB[localRow][localCol] = dataB[bRow * pc.N + col];
        } else {
            tileB[localRow][localCol] = (%SCALAR_DTYPE%)(0);
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute partial sum for this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write result
    if (row < pc.M && col < pc.N) {
        dataC[row * pc.N + col] = sum;
    }
}
