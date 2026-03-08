#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Linear-dispatch tiled matmul with register blocking: TILE_SIZE=%TILE_SIZE%, TM=%TM%, TN=%TN%
// Same algorithm as MatMulTiledReg but with 1D threadgroup (TILE_SIZE*TILE_SIZE, 1, 1)
// instead of 2D (TILE_SIZE, TILE_SIZE, 1). Thread derives row/col via div/mod.
// Optimizations: shared memory padding, K-unroll by 4 with mad(), bounds check hoisting

#define TILE_SIZE %TILE_SIZE%
#define TM %TM%
#define TN %TN%

#include "MatMulCommon.shaderh"

// +1 padding on inner dimension to avoid shared memory bank conflicts
groupshared %SCALAR_DTYPE_INPUT1% tileA[TILE_SIZE * TM][TILE_SIZE + 1];
groupshared %SCALAR_DTYPE_INPUT2% tileB[TILE_SIZE][TILE_SIZE * TN + 1];

[numthreads(TILE_SIZE * TILE_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint localRow = GTid.x / TILE_SIZE;
    uint localCol = GTid.x % TILE_SIZE;

    uint blockRowStart = Gid.y * TILE_SIZE * TM;
    uint blockColStart = Gid.x * TILE_SIZE * TN;

    %SCALAR_DTYPE_OUTPUT% acc[TM][TN];
    [unroll] for (uint m = 0; m < TM; m++)
        [unroll] for (uint n = 0; n < TN; n++)
            acc[m][n] = (%SCALAR_DTYPE_OUTPUT%)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;
    uint fullKTiles = pc.K / TILE_SIZE;
    bool hasPartialK = (pc.K % TILE_SIZE) != 0;
    bool fullM = (blockRowStart + TILE_SIZE * TM <= pc.M);
    bool fullN = (blockColStart + TILE_SIZE * TN <= pc.N);

    if (fullM && fullN) {
        // Fast path: interior workgroup — skip bounds checks on loads and output
        for (uint t = 0; t < fullKTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            [unroll] for (uint m = 0; m < TM; m++) {
                tileA[localRow + m * TILE_SIZE][localCol] =
                    loadA_fast(blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            }
            [unroll] for (uint n = 0; n < TN; n++) {
                tileB[localRow][localCol + n * TILE_SIZE] =
                    loadB_fast(tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
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

        // Partial K tile (last tile, K not multiple of TILE_SIZE): use checked loads
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

        // Unchecked output write — all outputs in bounds for interior workgroup
        [unroll] for (uint m = 0; m < TM; m++) {
            [unroll] for (uint n = 0; n < TN; n++) {
                uint outRow = blockRowStart + localRow + m * TILE_SIZE;
                uint outCol = blockColStart + localCol + n * TILE_SIZE;
                writeOutput(outRow, outCol, acc[m][n]);
            }
        }
    } else {
        // Edge workgroup: use checked loads and output bounds checks
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

        // Checked output write — edge workgroup needs bounds checks
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
