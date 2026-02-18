#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Tiled matmul: TILE_SIZE=16, TM=8, TN=8
// 256 threads/WG, effective 128x128 tile, 16KB shared memory
// High register pressure: 64 accumulators + 16 loads per thread per K step

#define TILE_SIZE 16
#define TM 8
#define TN 8

struct PushConstants {
    uint M;
    uint K;
    uint N;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataA;
[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataB;
[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataC;

groupshared %SCALAR_DTYPE% tileA[TILE_SIZE * TM][TILE_SIZE];
groupshared %SCALAR_DTYPE% tileB[TILE_SIZE][TILE_SIZE * TN];

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
            tileA[localRow + m * TILE_SIZE][localCol] =
                (aRow < pc.M && aCol < pc.K) ? dataA[aRow * pc.K + aCol] : (%SCALAR_DTYPE%)(0);
        }

        [unroll] for (uint n = 0; n < TN; n++) {
            uint bRow = tileKStart + localRow;
            uint bCol = blockColStart + localCol + n * TILE_SIZE;
            tileB[localRow][localCol + n * TILE_SIZE] =
                (bRow < pc.K && bCol < pc.N) ? dataB[bRow * pc.N + bCol] : (%SCALAR_DTYPE%)(0);
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
                dataC[outRow * pc.N + outCol] = acc[m][n];
            }
        }
    }
}
