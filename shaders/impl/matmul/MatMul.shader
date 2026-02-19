#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

#define TILE_SIZE 16

struct PushConstants {
    uint M;  // rows of A
    uint K;  // cols of A / rows of B
    uint N;  // cols of B
    uint strideA; // padded K (multiple of 4)
    uint strideB; // padded N (multiple of 4)
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE%> dataA;
[[vk::binding(1, 0)]] StructuredBuffer<%VEC_DTYPE%> dataB;
[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataC;

groupshared %SCALAR_DTYPE% tileA[TILE_SIZE][TILE_SIZE];
groupshared %SCALAR_DTYPE% tileB[TILE_SIZE][TILE_SIZE];

%SCALAR_DTYPE% loadA(uint row, uint col) {
    if (row >= pc.M || col >= pc.K) return (%SCALAR_DTYPE%)(0);
    uint idx = row * pc.strideA + col;
    return dataA[idx >> 2][idx & 3];
}

%SCALAR_DTYPE% loadB(uint row, uint col) {
    if (row >= pc.K || col >= pc.N) return (%SCALAR_DTYPE%)(0);
    uint idx = row * pc.strideB + col;
    return dataB[idx >> 2][idx & 3];
}

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
