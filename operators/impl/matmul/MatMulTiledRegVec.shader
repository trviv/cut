#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Tiled matmul with vec4 cooperative global memory loads
// TILE_SIZE=%TILE_SIZE%, TM=%TM%, TN=%TN%
// Optimization: threads cooperatively load full vec4s from global memory
// instead of extracting individual scalars. Reduces global load instructions
// from TM+TN per thread to 2 vec4 loads per thread (for T16R4x4).
// Requires: TILE_SIZE * TM * TILE_SIZE / 4 == TILE_SIZE^2 (threads)
//           TILE_SIZE * TILE_SIZE * TN / 4 == TILE_SIZE^2 (threads)
//           i.e. TM == 4 and TN == 4

#define TILE_SIZE %TILE_SIZE%
#define TM %TM%
#define TN %TN%

#include "MatMulCommon.shaderh"

groupshared %SCALAR_DTYPE_INPUT1% tileA[TILE_SIZE * TM][TILE_SIZE + 1];
groupshared %SCALAR_DTYPE_INPUT2% tileB[TILE_SIZE][TILE_SIZE * TN + 1];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint localRow = GTid.y;
    uint localCol = GTid.x;
    uint tid = localRow * TILE_SIZE + localCol;

    uint2 tileId = swizzleTileId(Gid);
    uint blockRowStart = tileId.y * TILE_SIZE * TM;
    uint blockColStart = tileId.x * TILE_SIZE * TN;

    %SCALAR_DTYPE_OUTPUT% acc[TM][TN];
    [unroll] for (uint m = 0; m < TM; m++)
        [unroll] for (uint n = 0; n < TN; n++)
            acc[m][n] = (%SCALAR_DTYPE_OUTPUT%)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;
    uint fullKTiles = pc.K / TILE_SIZE;
    bool hasPartialK = (pc.K % TILE_SIZE) != 0;
    bool fullM = (blockRowStart + TILE_SIZE * TM <= pc.M);
    bool fullN = (blockColStart + TILE_SIZE * TN <= pc.N);

    // Vec4 cooperative load mappings (precomputed)
    // tileA: [TILE_SIZE*TM][TILE_SIZE] = 64x16 = 1024 scalars = 256 vec4s
    uint tileA_row = tid / (TILE_SIZE / 4);   // tid / 4, range 0..63
    uint tileA_v4  = tid % (TILE_SIZE / 4);   // tid % 4, range 0..3
    uint tileA_col = tileA_v4 * 4;            // 0, 4, 8, 12

    // tileB: [TILE_SIZE][TILE_SIZE*TN] = 16x64 = 1024 scalars = 256 vec4s
    uint tileB_row = tid / (TILE_SIZE * TN / 4);  // tid / 16, range 0..15
    uint tileB_v4  = tid % (TILE_SIZE * TN / 4);  // tid % 16, range 0..15
    uint tileB_col = tileB_v4 * 4;                // 0, 4, 8, ..., 60

    if (fullM && fullN) {
        for (uint t = 0; t < fullKTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            // Vec4 load for tileA: 1 vec4 per thread
            uint gA_row = blockRowStart + tileA_row;
            uint gA_idx = gA_row * pc.strideA + tileKStart + tileA_col;
            %VEC_DTYPE_INPUT1% vA = dataA[gA_idx >> 2];
            tileA[tileA_row][tileA_col + 0] = vA[0];
            tileA[tileA_row][tileA_col + 1] = vA[1];
            tileA[tileA_row][tileA_col + 2] = vA[2];
            tileA[tileA_row][tileA_col + 3] = vA[3];

            // Vec4 load for tileB: 1 vec4 per thread
            uint gB_row = tileKStart + tileB_row;
            uint gB_idx = gB_row * pc.strideB + blockColStart + tileB_col;
            %VEC_DTYPE_INPUT2% vB = dataB[gB_idx >> 2];
            tileB[tileB_row][tileB_col + 0] = vB[0];
            tileB[tileB_row][tileB_col + 1] = vB[1];
            tileB[tileB_row][tileB_col + 2] = vB[2];
            tileB[tileB_row][tileB_col + 3] = vB[3];

            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                [unroll] for (uint m = 0; m < TM; m++) {
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 3];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        uint bc = localCol + n * TILE_SIZE;
                        acc[m][n] = mad(a0, (%SCALAR_DTYPE_OUTPUT%)tileB[k    ][bc], acc[m][n]);
                        acc[m][n] = mad(a1, (%SCALAR_DTYPE_OUTPUT%)tileB[k + 1][bc], acc[m][n]);
                        acc[m][n] = mad(a2, (%SCALAR_DTYPE_OUTPUT%)tileB[k + 2][bc], acc[m][n]);
                        acc[m][n] = mad(a3, (%SCALAR_DTYPE_OUTPUT%)tileB[k + 3][bc], acc[m][n]);
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Partial K tile: fall back to scalar checked loads
        if (hasPartialK) {
            uint tileKStart = fullKTiles * TILE_SIZE;

            [unroll] for (uint m = 0; m < TM; m++) {
                tileA[localRow + m * TILE_SIZE][localCol] =
                    loadA(blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            }
            [unroll] for (uint n = 0; n < TN; n++) {
                tileB[localRow][localCol + n * TILE_SIZE] =
                    loadB(tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }

            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                [unroll] for (uint m = 0; m < TM; m++) {
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 3];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        uint bc = localCol + n * TILE_SIZE;
                        acc[m][n] = mad(a0, (%SCALAR_DTYPE_OUTPUT%)tileB[k    ][bc], acc[m][n]);
                        acc[m][n] = mad(a1, (%SCALAR_DTYPE_OUTPUT%)tileB[k + 1][bc], acc[m][n]);
                        acc[m][n] = mad(a2, (%SCALAR_DTYPE_OUTPUT%)tileB[k + 2][bc], acc[m][n]);
                        acc[m][n] = mad(a3, (%SCALAR_DTYPE_OUTPUT%)tileB[k + 3][bc], acc[m][n]);
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        [unroll] for (uint m = 0; m < TM; m++) {
            [unroll] for (uint n = 0; n < TN; n++) {
                uint outRow = blockRowStart + localRow + m * TILE_SIZE;
                uint outCol = blockColStart + localCol + n * TILE_SIZE;
                writeOutput(outRow, outCol, acc[m][n]);
            }
        }
    } else {
        // Edge workgroup: scalar checked loads throughout
        for (uint t = 0; t < numTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            [unroll] for (uint m = 0; m < TM; m++) {
                tileA[localRow + m * TILE_SIZE][localCol] =
                    loadA(blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            }
            [unroll] for (uint n = 0; n < TN; n++) {
                tileB[localRow][localCol + n * TILE_SIZE] =
                    loadB(tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }

            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                [unroll] for (uint m = 0; m < TM; m++) {
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 3];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        uint bc = localCol + n * TILE_SIZE;
                        acc[m][n] = mad(a0, (%SCALAR_DTYPE_OUTPUT%)tileB[k    ][bc], acc[m][n]);
                        acc[m][n] = mad(a1, (%SCALAR_DTYPE_OUTPUT%)tileB[k + 1][bc], acc[m][n]);
                        acc[m][n] = mad(a2, (%SCALAR_DTYPE_OUTPUT%)tileB[k + 2][bc], acc[m][n]);
                        acc[m][n] = mad(a3, (%SCALAR_DTYPE_OUTPUT%)tileB[k + 3][bc], acc[m][n]);
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
                    writeOutput(outRow, outCol, acc[m][n]);
                }
            }
        }
    }
}
