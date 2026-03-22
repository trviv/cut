#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Aligned variant of tiled matmul with vec4 loads + B-register pre-load.
// TILE_SIZE=%TILE_SIZE%, TM=%TM%, TN=%TN%
//
// Optimized for dimensions where K % TILE_SIZE == 0 and N % (TILE_SIZE*TN) == 0.
// This eliminates:
//   1. Partial K tile handling (no scalar load fallback)
//   2. N-edge workgroups (fullN is always true)
//   3. Scalar loads in edge workgroups for B matrix
// Only M-edge workgroups need special handling (vec4 loads with row check).

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

    // K % TILE_SIZE == 0 guaranteed: exact number of tiles, no partial
    uint numTiles = pc.K / TILE_SIZE;

    // N % (TILE_SIZE*TN) == 0 guaranteed: fullN is always true
    bool fullM = (blockRowStart + TILE_SIZE * TM <= pc.M);

    // Vec4 cooperative load mappings
    uint tileA_row = tid / (TILE_SIZE / 4);
    uint tileA_col = (tid % (TILE_SIZE / 4)) * 4;

    uint tileB_row = tid / (TILE_SIZE * TN / 4);
    uint tileB_col = (tid % (TILE_SIZE * TN / 4)) * 4;

    if (fullM) {
        // Fast path: all dimensions aligned, no bounds checks needed
        for (uint t = 0; t < numTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            // Vec4 load for tileA — no bounds check
            uint gA_idx = (blockRowStart + tileA_row) * pc.strideA + tileKStart + tileA_col;
            %VEC_DTYPE_INPUT1% vA = dataA[gA_idx >> 2];
            tileA[tileA_row][tileA_col + 0] = vA[0];
            tileA[tileA_row][tileA_col + 1] = vA[1];
            tileA[tileA_row][tileA_col + 2] = vA[2];
            tileA[tileA_row][tileA_col + 3] = vA[3];

            // Vec4 load for tileB — no bounds check
            uint gB_idx = (tileKStart + tileB_row) * pc.strideB + blockColStart + tileB_col;
            %VEC_DTYPE_INPUT2% vB = dataB[gB_idx >> 2];
            tileB[tileB_row][tileB_col + 0] = vB[0];
            tileB[tileB_row][tileB_col + 1] = vB[1];
            tileB[tileB_row][tileB_col + 2] = vB[2];
            tileB[tileB_row][tileB_col + 3] = vB[3];

            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                %SCALAR_DTYPE_OUTPUT% b_reg[4][TN];
                [unroll] for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k    ][bc];
                    b_reg[1][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 1][bc];
                    b_reg[2][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 2][bc];
                    b_reg[3][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 3][bc];
                }
                [unroll] for (uint m = 0; m < TM; m++) {
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 3];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Write output — no bounds check needed
        [unroll] for (uint m = 0; m < TM; m++) {
            [unroll] for (uint n = 0; n < TN; n++) {
                uint outRow = blockRowStart + localRow + m * TILE_SIZE;
                uint outCol = blockColStart + localCol + n * TILE_SIZE;
                writeOutput(outRow, outCol, acc[m][n]);
            }
        }
    } else {
        // M-edge workgroup: only M can be out of bounds.
        // K and N are aligned, so B loads are always safe with vec4.
        // A loads use vec4 with a per-row bounds check (not per-element).
        for (uint t = 0; t < numTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            // Vec4 load for tileA — row bounds check only
            uint gA_row = blockRowStart + tileA_row;
            if (gA_row < pc.M) {
                uint gA_idx = gA_row * pc.strideA + tileKStart + tileA_col;
                %VEC_DTYPE_INPUT1% vA = dataA[gA_idx >> 2];
                tileA[tileA_row][tileA_col + 0] = vA[0];
                tileA[tileA_row][tileA_col + 1] = vA[1];
                tileA[tileA_row][tileA_col + 2] = vA[2];
                tileA[tileA_row][tileA_col + 3] = vA[3];
            } else {
                tileA[tileA_row][tileA_col + 0] = (%SCALAR_DTYPE_INPUT1%)(0);
                tileA[tileA_row][tileA_col + 1] = (%SCALAR_DTYPE_INPUT1%)(0);
                tileA[tileA_row][tileA_col + 2] = (%SCALAR_DTYPE_INPUT1%)(0);
                tileA[tileA_row][tileA_col + 3] = (%SCALAR_DTYPE_INPUT1%)(0);
            }

            // Vec4 load for tileB — always safe (K and N aligned)
            uint gB_idx = (tileKStart + tileB_row) * pc.strideB + blockColStart + tileB_col;
            %VEC_DTYPE_INPUT2% vB = dataB[gB_idx >> 2];
            tileB[tileB_row][tileB_col + 0] = vB[0];
            tileB[tileB_row][tileB_col + 1] = vB[1];
            tileB[tileB_row][tileB_col + 2] = vB[2];
            tileB[tileB_row][tileB_col + 3] = vB[3];

            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                %SCALAR_DTYPE_OUTPUT% b_reg[4][TN];
                [unroll] for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k    ][bc];
                    b_reg[1][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 1][bc];
                    b_reg[2][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 2][bc];
                    b_reg[3][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 3][bc];
                }
                [unroll] for (uint m = 0; m < TM; m++) {
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 3];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Write output — row bounds check only (N always in bounds)
        [unroll] for (uint m = 0; m < TM; m++) {
            uint outRow = blockRowStart + localRow + m * TILE_SIZE;
            if (outRow < pc.M) {
                [unroll] for (uint n = 0; n < TN; n++) {
                    uint outCol = blockColStart + localCol + n * TILE_SIZE;
                    writeOutput(outRow, outCol, acc[m][n]);
                }
            }
        }
    }
}
