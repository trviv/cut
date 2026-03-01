#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

#define TILE_SIZE 16

#include "MatMulCommon.shaderh"

groupshared %SCALAR_DTYPE_INPUT1% tileA[TILE_SIZE][TILE_SIZE];
groupshared %SCALAR_DTYPE_INPUT2% tileB[TILE_SIZE][TILE_SIZE];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID) {
    uint row = DTid.y;
    uint col = DTid.x;
    uint localRow = GTid.y;
    uint localCol = GTid.x;

    %SCALAR_DTYPE_OUTPUT% sum = (%SCALAR_DTYPE_OUTPUT%)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + localCol;
        tileA[localRow][localCol] = loadA(row, aCol);

        uint bRow = t * TILE_SIZE + localRow;
        tileB[localRow][localCol] = loadB(bRow, col);

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += (%SCALAR_DTYPE_OUTPUT%)(tileA[localRow][k]) * (%SCALAR_DTYPE_OUTPUT%)(tileB[k][localCol]);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < pc.M && col < pc.N) {
        dataC[row * pc.strideB + col] = sum;
    }
}
