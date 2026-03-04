#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Fused tiled matmul with SiLU activation: silu(A * B)
// where silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
// Tiling parameters: TILE_SIZE=%TILE_SIZE%, TM=%TM%, TN=%TN%

#define TILE_SIZE %TILE_SIZE%
#define TM %TM%
#define TN %TN%

#include "MatMulCommon.shaderh"

groupshared %SCALAR_DTYPE_INPUT1% tileA[TILE_SIZE * TM][TILE_SIZE];
groupshared %SCALAR_DTYPE_INPUT2% tileB[TILE_SIZE][TILE_SIZE * TN];

// SiLU activation: x / (1 + exp(-x)) = x * sigmoid(x)
%SCALAR_DTYPE_OUTPUT% silu(%SCALAR_DTYPE_OUTPUT% x) {
    return x / ((%SCALAR_DTYPE_OUTPUT%)(1.0) + exp(-x));
}

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

    // Matmul computation (identical to standard matmul)
    for (uint t = 0; t < numTiles; t++) {
        uint tileKStart = t * TILE_SIZE;

        // Load tile of A into shared memory
        [unroll] for (uint m = 0; m < TM; m++) {
            uint aRow = blockRowStart + localRow + m * TILE_SIZE;
            uint aCol = tileKStart + localCol;
            tileA[localRow + m * TILE_SIZE][localCol] = loadA(aRow, aCol);
        }

        // Load tile of B into shared memory
        [unroll] for (uint n = 0; n < TN; n++) {
            uint bRow = tileKStart + localRow;
            uint bCol = blockColStart + localCol + n * TILE_SIZE;
            tileB[localRow][localCol + n * TILE_SIZE] = loadB(bRow, bCol);
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute tile contributions
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

    // Write output with SiLU activation applied inline
    [unroll] for (uint m = 0; m < TM; m++) {
        [unroll] for (uint n = 0; n < TN; n++) {
            uint outRow = blockRowStart + localRow + m * TILE_SIZE;
            uint outCol = blockColStart + localCol + n * TILE_SIZE;
            if (outRow < pc.M && outCol < pc.N) {
                // Apply SiLU activation before writing
                dataC[outRow * pc.strideB + outCol] = silu(acc[m][n]);
            }
        }
    }
}
