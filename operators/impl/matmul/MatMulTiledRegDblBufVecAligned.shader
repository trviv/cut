#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Double-buffered + Vec4 aligned tiled matmul with K-unroll by 8.
// TILE_SIZE=%TILE_SIZE%, TM=%TM%, TN=%TN%
//
// Combines the best optimizations:
//   1. Double buffering — load next tile while computing current (1 barrier vs 2)
//   2. Vec4 cooperative loads — 4x fewer global load instructions
//   3. K-unroll by 8 with A+B register pre-load — maximum ILP
//   4. Aligned constraints (K%16==0, N%64==0) — eliminate all edge handling
//
// Requirements: K % TILE_SIZE == 0 and N % (TILE_SIZE*TN) == 0

#define TILE_SIZE %TILE_SIZE%
#define TM %TM%
#define TN %TN%

#include "MatMulCommon.shaderh"

// Double-buffered shared memory: [2] for ping-pong
groupshared %SCALAR_DTYPE_INPUT1% tileA[2][TILE_SIZE * TM][TILE_SIZE + 1];
groupshared %SCALAR_DTYPE_INPUT2% tileB[2][TILE_SIZE][TILE_SIZE * TN + 1];

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

    // K % TILE_SIZE == 0 guaranteed
    uint numTiles = pc.K / TILE_SIZE;
    // N % (TILE_SIZE*TN) == 0 guaranteed: fullN is always true
    bool fullM = (blockRowStart + TILE_SIZE * TM <= pc.M);

    // Vec4 cooperative load mappings
    uint tileA_row = tid / (TILE_SIZE / 4);
    uint tileA_col = (tid % (TILE_SIZE / 4)) * 4;
    uint tileB_row = tid / (TILE_SIZE * TN / 4);
    uint tileB_col = (tid % (TILE_SIZE * TN / 4)) * 4;

    if (numTiles == 0) {
        // Nothing to compute
    } else if (fullM) {
        // ============================================================
        // Fast path: all dimensions aligned, no bounds checks
        // ============================================================

        // Prefetch first tile into buffer 0 (vec4 loads)
        {
            uint gA_idx = (blockRowStart + tileA_row) * pc.strideA + tileA_col;
            %VEC_DTYPE_INPUT1% vA = dataA[gA_idx >> 2];
            tileA[0][tileA_row][tileA_col + 0] = vA[0];
            tileA[0][tileA_row][tileA_col + 1] = vA[1];
            tileA[0][tileA_row][tileA_col + 2] = vA[2];
            tileA[0][tileA_row][tileA_col + 3] = vA[3];

            uint gB_idx = tileB_row * pc.strideB + blockColStart + tileB_col;
            %VEC_DTYPE_INPUT2% vB = dataB[gB_idx >> 2];
            tileB[0][tileB_row][tileB_col + 0] = vB[0];
            tileB[0][tileB_row][tileB_col + 1] = vB[1];
            tileB[0][tileB_row][tileB_col + 2] = vB[2];
            tileB[0][tileB_row][tileB_col + 3] = vB[3];
        }
        GroupMemoryBarrierWithGroupSync();

        // Double-buffered loop: load tile t into buf[t&1], compute buf[(t-1)&1]
        for (uint t = 1; t < numTiles; t++) {
            uint cur = (t - 1) & 1;
            uint nxt = t & 1;
            uint tileKStart = t * TILE_SIZE;

            // Vec4 load next tile into nxt buffer
            uint gA_idx = (blockRowStart + tileA_row) * pc.strideA + tileKStart + tileA_col;
            %VEC_DTYPE_INPUT1% vA = dataA[gA_idx >> 2];
            tileA[nxt][tileA_row][tileA_col + 0] = vA[0];
            tileA[nxt][tileA_row][tileA_col + 1] = vA[1];
            tileA[nxt][tileA_row][tileA_col + 2] = vA[2];
            tileA[nxt][tileA_row][tileA_col + 3] = vA[3];

            uint gB_idx = (tileKStart + tileB_row) * pc.strideB + blockColStart + tileB_col;
            %VEC_DTYPE_INPUT2% vB = dataB[gB_idx >> 2];
            tileB[nxt][tileB_row][tileB_col + 0] = vB[0];
            tileB[nxt][tileB_row][tileB_col + 1] = vB[1];
            tileB[nxt][tileB_row][tileB_col + 2] = vB[2];
            tileB[nxt][tileB_row][tileB_col + 3] = vB[3];

            // Compute on cur buffer with K-unroll by 8
            [unroll] for (uint k = 0; k < TILE_SIZE; k += 8) {
                %SCALAR_DTYPE_OUTPUT% b_reg[8][TN];
                [unroll] for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k    ][bc];
                    b_reg[1][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 1][bc];
                    b_reg[2][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 2][bc];
                    b_reg[3][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 3][bc];
                    b_reg[4][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 4][bc];
                    b_reg[5][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 5][bc];
                    b_reg[6][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 6][bc];
                    b_reg[7][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 7][bc];
                }
                [unroll] for (uint m = 0; m < TM; m++) {
                    uint ar = localRow + m * TILE_SIZE;
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 3];
                    %SCALAR_DTYPE_OUTPUT% a4 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 4];
                    %SCALAR_DTYPE_OUTPUT% a5 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 5];
                    %SCALAR_DTYPE_OUTPUT% a6 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 6];
                    %SCALAR_DTYPE_OUTPUT% a7 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 7];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
                        acc[m][n] = mad(a4, b_reg[4][n], acc[m][n]);
                        acc[m][n] = mad(a5, b_reg[5][n], acc[m][n]);
                        acc[m][n] = mad(a6, b_reg[6][n], acc[m][n]);
                        acc[m][n] = mad(a7, b_reg[7][n], acc[m][n]);
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Compute last tile
        {
            uint lastBuf = (numTiles - 1) & 1;
            [unroll] for (uint k = 0; k < TILE_SIZE; k += 8) {
                %SCALAR_DTYPE_OUTPUT% b_reg[8][TN];
                [unroll] for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k    ][bc];
                    b_reg[1][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 1][bc];
                    b_reg[2][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 2][bc];
                    b_reg[3][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 3][bc];
                    b_reg[4][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 4][bc];
                    b_reg[5][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 5][bc];
                    b_reg[6][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 6][bc];
                    b_reg[7][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 7][bc];
                }
                [unroll] for (uint m = 0; m < TM; m++) {
                    uint ar = localRow + m * TILE_SIZE;
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 3];
                    %SCALAR_DTYPE_OUTPUT% a4 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 4];
                    %SCALAR_DTYPE_OUTPUT% a5 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 5];
                    %SCALAR_DTYPE_OUTPUT% a6 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 6];
                    %SCALAR_DTYPE_OUTPUT% a7 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 7];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
                        acc[m][n] = mad(a4, b_reg[4][n], acc[m][n]);
                        acc[m][n] = mad(a5, b_reg[5][n], acc[m][n]);
                        acc[m][n] = mad(a6, b_reg[6][n], acc[m][n]);
                        acc[m][n] = mad(a7, b_reg[7][n], acc[m][n]);
                    }
                }
            }
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
        // ============================================================
        // M-edge: row bounds check on A loads only
        // ============================================================

        // Prefetch first tile into buffer 0
        {
            uint gA_row = blockRowStart + tileA_row;
            if (gA_row < pc.M) {
                uint gA_idx = gA_row * pc.strideA + tileA_col;
                %VEC_DTYPE_INPUT1% vA = dataA[gA_idx >> 2];
                tileA[0][tileA_row][tileA_col + 0] = vA[0];
                tileA[0][tileA_row][tileA_col + 1] = vA[1];
                tileA[0][tileA_row][tileA_col + 2] = vA[2];
                tileA[0][tileA_row][tileA_col + 3] = vA[3];
            } else {
                tileA[0][tileA_row][tileA_col + 0] = (%SCALAR_DTYPE_INPUT1%)(0);
                tileA[0][tileA_row][tileA_col + 1] = (%SCALAR_DTYPE_INPUT1%)(0);
                tileA[0][tileA_row][tileA_col + 2] = (%SCALAR_DTYPE_INPUT1%)(0);
                tileA[0][tileA_row][tileA_col + 3] = (%SCALAR_DTYPE_INPUT1%)(0);
            }

            uint gB_idx = tileB_row * pc.strideB + blockColStart + tileB_col;
            %VEC_DTYPE_INPUT2% vB = dataB[gB_idx >> 2];
            tileB[0][tileB_row][tileB_col + 0] = vB[0];
            tileB[0][tileB_row][tileB_col + 1] = vB[1];
            tileB[0][tileB_row][tileB_col + 2] = vB[2];
            tileB[0][tileB_row][tileB_col + 3] = vB[3];
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint t = 1; t < numTiles; t++) {
            uint cur = (t - 1) & 1;
            uint nxt = t & 1;
            uint tileKStart = t * TILE_SIZE;

            // Load next tile — A with row check, B unchecked
            uint gA_row = blockRowStart + tileA_row;
            if (gA_row < pc.M) {
                uint gA_idx = gA_row * pc.strideA + tileKStart + tileA_col;
                %VEC_DTYPE_INPUT1% vA = dataA[gA_idx >> 2];
                tileA[nxt][tileA_row][tileA_col + 0] = vA[0];
                tileA[nxt][tileA_row][tileA_col + 1] = vA[1];
                tileA[nxt][tileA_row][tileA_col + 2] = vA[2];
                tileA[nxt][tileA_row][tileA_col + 3] = vA[3];
            } else {
                tileA[nxt][tileA_row][tileA_col + 0] = (%SCALAR_DTYPE_INPUT1%)(0);
                tileA[nxt][tileA_row][tileA_col + 1] = (%SCALAR_DTYPE_INPUT1%)(0);
                tileA[nxt][tileA_row][tileA_col + 2] = (%SCALAR_DTYPE_INPUT1%)(0);
                tileA[nxt][tileA_row][tileA_col + 3] = (%SCALAR_DTYPE_INPUT1%)(0);
            }

            uint gB_idx = (tileKStart + tileB_row) * pc.strideB + blockColStart + tileB_col;
            %VEC_DTYPE_INPUT2% vB = dataB[gB_idx >> 2];
            tileB[nxt][tileB_row][tileB_col + 0] = vB[0];
            tileB[nxt][tileB_row][tileB_col + 1] = vB[1];
            tileB[nxt][tileB_row][tileB_col + 2] = vB[2];
            tileB[nxt][tileB_row][tileB_col + 3] = vB[3];

            // Compute on cur buffer
            [unroll] for (uint k = 0; k < TILE_SIZE; k += 8) {
                %SCALAR_DTYPE_OUTPUT% b_reg[8][TN];
                [unroll] for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k    ][bc];
                    b_reg[1][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 1][bc];
                    b_reg[2][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 2][bc];
                    b_reg[3][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 3][bc];
                    b_reg[4][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 4][bc];
                    b_reg[5][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 5][bc];
                    b_reg[6][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 6][bc];
                    b_reg[7][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 7][bc];
                }
                [unroll] for (uint m = 0; m < TM; m++) {
                    uint ar = localRow + m * TILE_SIZE;
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 3];
                    %SCALAR_DTYPE_OUTPUT% a4 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 4];
                    %SCALAR_DTYPE_OUTPUT% a5 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 5];
                    %SCALAR_DTYPE_OUTPUT% a6 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 6];
                    %SCALAR_DTYPE_OUTPUT% a7 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][ar][k + 7];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
                        acc[m][n] = mad(a4, b_reg[4][n], acc[m][n]);
                        acc[m][n] = mad(a5, b_reg[5][n], acc[m][n]);
                        acc[m][n] = mad(a6, b_reg[6][n], acc[m][n]);
                        acc[m][n] = mad(a7, b_reg[7][n], acc[m][n]);
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Compute last tile
        {
            uint lastBuf = (numTiles - 1) & 1;
            [unroll] for (uint k = 0; k < TILE_SIZE; k += 8) {
                %SCALAR_DTYPE_OUTPUT% b_reg[8][TN];
                [unroll] for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k    ][bc];
                    b_reg[1][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 1][bc];
                    b_reg[2][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 2][bc];
                    b_reg[3][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 3][bc];
                    b_reg[4][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 4][bc];
                    b_reg[5][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 5][bc];
                    b_reg[6][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 6][bc];
                    b_reg[7][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[lastBuf][k + 7][bc];
                }
                [unroll] for (uint m = 0; m < TM; m++) {
                    uint ar = localRow + m * TILE_SIZE;
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 3];
                    %SCALAR_DTYPE_OUTPUT% a4 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 4];
                    %SCALAR_DTYPE_OUTPUT% a5 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 5];
                    %SCALAR_DTYPE_OUTPUT% a6 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 6];
                    %SCALAR_DTYPE_OUTPUT% a7 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastBuf][ar][k + 7];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
                        acc[m][n] = mad(a4, b_reg[4][n], acc[m][n]);
                        acc[m][n] = mad(a5, b_reg[5][n], acc[m][n]);
                        acc[m][n] = mad(a6, b_reg[6][n], acc[m][n]);
                        acc[m][n] = mad(a7, b_reg[7][n], acc[m][n]);
                    }
                }
            }
        }

        // Write output — row bounds check only
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
