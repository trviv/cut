#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_OUTPUT%

// Q8_0 dequant matmul with register blocking: TILE_SIZE=%TILE_SIZE%, TM=%TM%, TN=%TN%

#define TILE_SIZE %TILE_SIZE%
#define TM %TM%
#define TN %TN%

#include "MatMulQ8Common.shaderh"

groupshared %SCALAR_DTYPE_INPUT1% tileA[TILE_SIZE * TM][TILE_SIZE];
groupshared float tileB[TILE_SIZE][TILE_SIZE * TN];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint localRow = GTid.y;
    uint localCol = GTid.x;

    uint blockRowStart = Gid.y * TILE_SIZE * TM;
    uint blockColStart = Gid.x * TILE_SIZE * TN;

    %SCALAR_DTYPE_OUTPUT% acc[TM][TN];
    [unroll] for (uint m = 0; m < TM; m++)
        [unroll] for (uint n = 0; n < TN; n++)
            acc[m][n] = (%SCALAR_DTYPE_OUTPUT%)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint tileKStart = t * TILE_SIZE;

        [unroll] for (uint m = 0; m < TM; m++) {
            uint aRow = blockRowStart + localRow + m * TILE_SIZE;
            uint aCol = tileKStart + localCol;
            tileA[localRow + m * TILE_SIZE][localCol] = loadA(aRow, aCol);
        }

        [unroll] for (uint n = 0; n < TN; n++) {
            uint bK = tileKStart + localRow;
            uint bN = blockColStart + localCol + n * TILE_SIZE;
            tileB[localRow][localCol + n * TILE_SIZE] = loadB(bK, bN);
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < TILE_SIZE; k++) {
            [unroll] for (uint m = 0; m < TM; m++) {
                %SCALAR_DTYPE_OUTPUT% aVal = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k];
                [unroll] for (uint n = 0; n < TN; n++) {
                    acc[m][n] += aVal * (%SCALAR_DTYPE_OUTPUT%)tileB[k][localCol + n * TILE_SIZE];
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
                dataC[outRow * pc.strideC + outCol] = acc[m][n];
            }
        }
    }
}
