#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

#define TILE_SIZE 16

#include "MatMulCommon.shaderh"

groupshared %SCALAR_DTYPE% tileA[TILE_SIZE][TILE_SIZE];
groupshared %SCALAR_DTYPE% tileB[TILE_SIZE][TILE_SIZE];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID) {
    uint row = DTid.y;
    uint col = DTid.x;
    uint localRow = GTid.y;
    uint localCol = GTid.x;

    %SCALAR_DTYPE% sum = (%SCALAR_DTYPE%)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + localCol;
        tileA[localRow][localCol] = loadA(row, aCol);

        uint bRow = t * TILE_SIZE + localRow;
        tileB[localRow][localCol] = loadB(bRow, col);

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < pc.M && col < pc.N) {
        dataC[row * pc.strideB + col] = sum;
    }
}
