#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Tiled matmul with register blocking: TILE_SIZE=8, TM=2, TN=2

#define TILE_SIZE 8
#define TM 2
#define TN 2

struct PushConstants {
    uint M;
    uint K;
    uint N;
    uint strideA; // padded K (multiple of 4)
    uint strideB; // padded N (multiple of 4)
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE%> dataA;
[[vk::binding(1, 0)]] StructuredBuffer<%VEC_DTYPE%> dataB;
[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataC;

groupshared %SCALAR_DTYPE% tileA[TILE_SIZE * TM][TILE_SIZE];
groupshared %SCALAR_DTYPE% tileB[TILE_SIZE][TILE_SIZE * TN];

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
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint localRow = GTid.y;
    uint localCol = GTid.x;

    uint blockRowStart = Gid.y * TILE_SIZE * TM;
    uint blockColStart = Gid.x * TILE_SIZE * TN;

    %SCALAR_DTYPE% acc[TM][TN];
    [unroll] for (uint m = 0; m < TM; m++)
        [unroll] for (uint n = 0; n < TN; n++)
            acc[m][n] = (%SCALAR_DTYPE%)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint tileKStart = t * TILE_SIZE;

        [unroll] for (uint m = 0; m < TM; m++) {
            uint aRow = blockRowStart + localRow + m * TILE_SIZE;
            uint aCol = tileKStart + localCol;
            tileA[localRow + m * TILE_SIZE][localCol] = loadA(aRow, aCol);
        }

        [unroll] for (uint n = 0; n < TN; n++) {
            uint bRow = tileKStart + localRow;
            uint bCol = blockColStart + localCol + n * TILE_SIZE;
            tileB[localRow][localCol + n * TILE_SIZE] = loadB(bRow, bCol);
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < TILE_SIZE; k++) {
            [unroll] for (uint m = 0; m < TM; m++) {
                %SCALAR_DTYPE% aVal = tileA[localRow + m * TILE_SIZE][k];
                [unroll] for (uint n = 0; n < TN; n++) {
                    acc[m][n] += aVal * tileB[k][localCol + n * TILE_SIZE];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint m = 0; m < TM; m++) {
        [unroll] for (uint n = 0; n < TN; n++) {
            uint outRow = blockRowStart + localRow + m * TILE_SIZE;
            uint outCol = blockColStart + localCol + n * TILE_SIZE;
            if (outRow < pc.M && outCol < pc.N) {
                dataC[outRow * pc.strideB + outCol] = acc[m][n];
            }
        }
    }
}
