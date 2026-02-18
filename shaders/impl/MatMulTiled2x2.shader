#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Tiled matrix multiplication with 2x2 register blocking per thread.
// Combines shared memory tiling (cooperative loading) with per-thread
// register blocking: each thread computes a 2x2 sub-tile of the output.
// Effective tile: 32x32 output per workgroup (16 threads * 2 outputs each).

#define TILE_SIZE 16
#define TM 2
#define TN 2

struct PushConstants {
    uint M;  // rows of A
    uint K;  // cols of A / rows of B
    uint N;  // cols of B
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataA;

[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataB;

[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataC;

// Shared memory tiles sized to cover the expanded output region
groupshared %SCALAR_DTYPE% tileA[TILE_SIZE * TM][TILE_SIZE];
groupshared %SCALAR_DTYPE% tileB[TILE_SIZE][TILE_SIZE * TN];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint localRow = GTid.y;
    uint localCol = GTid.x;

    // Output region covered by this workgroup
    uint blockRowStart = Gid.y * TILE_SIZE * TM;
    uint blockColStart = Gid.x * TILE_SIZE * TN;

    // Accumulator registers for TM x TN output per thread
    %SCALAR_DTYPE% acc[TM][TN];
    [unroll] for (uint m = 0; m < TM; m++)
        [unroll] for (uint n = 0; n < TN; n++)
            acc[m][n] = (%SCALAR_DTYPE%)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint tileKStart = t * TILE_SIZE;

        // Cooperative load: each thread loads TM elements of A
        [unroll] for (uint m = 0; m < TM; m++) {
            uint aRow = blockRowStart + localRow + m * TILE_SIZE;
            uint aCol = tileKStart + localCol;
            tileA[localRow + m * TILE_SIZE][localCol] =
                (aRow < pc.M && aCol < pc.K) ? dataA[aRow * pc.K + aCol] : (%SCALAR_DTYPE%)(0);
        }

        // Cooperative load: each thread loads TN elements of B
        [unroll] for (uint n = 0; n < TN; n++) {
            uint bRow = tileKStart + localRow;
            uint bCol = blockColStart + localCol + n * TILE_SIZE;
            tileB[localRow][localCol + n * TILE_SIZE] =
                (bRow < pc.K && bCol < pc.N) ? dataB[bRow * pc.N + bCol] : (%SCALAR_DTYPE%)(0);
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute: each thread accumulates TM x TN outputs from shared tiles
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

    // Write results
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
