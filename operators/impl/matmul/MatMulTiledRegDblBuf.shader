#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Tiled matmul with double-buffered shared memory
// TILE_SIZE=%TILE_SIZE%, TM=%TM%, TN=%TN%
// Optimization: uses two sets of shared memory tiles (ping-pong).
// While computing on one buffer, the next K-tile is loaded into the other,
// reducing barriers from 2 to 1 per K-tile iteration.

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
        if (fullKTiles > 0) {
            // Prefetch first full tile into buffer 0
            {
                uint tileKStart = 0;
                [unroll] for (uint m = 0; m < TM; m++)
                    tileA[0][localRow + m * TILE_SIZE][localCol] =
                        loadA_fast(blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
                [unroll] for (uint n = 0; n < TN; n++)
                    tileB[0][localRow][localCol + n * TILE_SIZE] =
                        loadB_fast(tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }
            GroupMemoryBarrierWithGroupSync();

            // Double-buffered loop: load tile t into buf[t&1], compute tile t-1 from buf[(t-1)&1]
            for (uint t = 1; t < fullKTiles; t++) {
                uint cur = (t - 1) & 1;
                uint nxt = t & 1;
                uint tileKStart = t * TILE_SIZE;

                // Load next full tile into nxt buffer
                [unroll] for (uint m = 0; m < TM; m++)
                    tileA[nxt][localRow + m * TILE_SIZE][localCol] =
                        loadA_fast(blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
                [unroll] for (uint n = 0; n < TN; n++)
                    tileB[nxt][localRow][localCol + n * TILE_SIZE] =
                        loadB_fast(tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);

                // Compute on cur buffer (previous tile)
                [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                    [unroll] for (uint m = 0; m < TM; m++) {
                        %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][localRow + m * TILE_SIZE][k];
                        %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][localRow + m * TILE_SIZE][k + 1];
                        %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][localRow + m * TILE_SIZE][k + 2];
                        %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[cur][localRow + m * TILE_SIZE][k + 3];
                        [unroll] for (uint n = 0; n < TN; n++) {
                            uint bc = localCol + n * TILE_SIZE;
                            acc[m][n] = mad(a0, (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k    ][bc], acc[m][n]);
                            acc[m][n] = mad(a1, (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 1][bc], acc[m][n]);
                            acc[m][n] = mad(a2, (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 2][bc], acc[m][n]);
                            acc[m][n] = mad(a3, (%SCALAR_DTYPE_OUTPUT%)tileB[cur][k + 3][bc], acc[m][n]);
                        }
                    }
                }

                GroupMemoryBarrierWithGroupSync();
            }

            if (hasPartialK) {
                // Load partial tile while computing last full tile
                uint lastFull = (fullKTiles - 1) & 1;
                uint partBuf = fullKTiles & 1;
                uint partKStart = fullKTiles * TILE_SIZE;

                // Load partial tile (checked) into partBuf
                [unroll] for (uint m = 0; m < TM; m++)
                    tileA[partBuf][localRow + m * TILE_SIZE][localCol] =
                        loadA(blockRowStart + localRow + m * TILE_SIZE, partKStart + localCol);
                [unroll] for (uint n = 0; n < TN; n++)
                    tileB[partBuf][localRow][localCol + n * TILE_SIZE] =
                        loadB(partKStart + localRow, blockColStart + localCol + n * TILE_SIZE);

                // Compute last full tile from lastFull buffer
                [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                    [unroll] for (uint m = 0; m < TM; m++) {
                        %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastFull][localRow + m * TILE_SIZE][k];
                        %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastFull][localRow + m * TILE_SIZE][k + 1];
                        %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastFull][localRow + m * TILE_SIZE][k + 2];
                        %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastFull][localRow + m * TILE_SIZE][k + 3];
                        [unroll] for (uint n = 0; n < TN; n++) {
                            uint bc = localCol + n * TILE_SIZE;
                            acc[m][n] = mad(a0, (%SCALAR_DTYPE_OUTPUT%)tileB[lastFull][k    ][bc], acc[m][n]);
                            acc[m][n] = mad(a1, (%SCALAR_DTYPE_OUTPUT%)tileB[lastFull][k + 1][bc], acc[m][n]);
                            acc[m][n] = mad(a2, (%SCALAR_DTYPE_OUTPUT%)tileB[lastFull][k + 2][bc], acc[m][n]);
                            acc[m][n] = mad(a3, (%SCALAR_DTYPE_OUTPUT%)tileB[lastFull][k + 3][bc], acc[m][n]);
                        }
                    }
                }

                GroupMemoryBarrierWithGroupSync();

                // Compute partial tile
                [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                    [unroll] for (uint m = 0; m < TM; m++) {
                        %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[partBuf][localRow + m * TILE_SIZE][k];
                        %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[partBuf][localRow + m * TILE_SIZE][k + 1];
                        %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[partBuf][localRow + m * TILE_SIZE][k + 2];
                        %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[partBuf][localRow + m * TILE_SIZE][k + 3];
                        [unroll] for (uint n = 0; n < TN; n++) {
                            uint bc = localCol + n * TILE_SIZE;
                            acc[m][n] = mad(a0, (%SCALAR_DTYPE_OUTPUT%)tileB[partBuf][k    ][bc], acc[m][n]);
                            acc[m][n] = mad(a1, (%SCALAR_DTYPE_OUTPUT%)tileB[partBuf][k + 1][bc], acc[m][n]);
                            acc[m][n] = mad(a2, (%SCALAR_DTYPE_OUTPUT%)tileB[partBuf][k + 2][bc], acc[m][n]);
                            acc[m][n] = mad(a3, (%SCALAR_DTYPE_OUTPUT%)tileB[partBuf][k + 3][bc], acc[m][n]);
                        }
                    }
                }
            } else {
                // Compute last full tile (no partial tile)
                uint lastFull = (fullKTiles - 1) & 1;
                [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                    [unroll] for (uint m = 0; m < TM; m++) {
                        %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastFull][localRow + m * TILE_SIZE][k];
                        %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastFull][localRow + m * TILE_SIZE][k + 1];
                        %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastFull][localRow + m * TILE_SIZE][k + 2];
                        %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[lastFull][localRow + m * TILE_SIZE][k + 3];
                        [unroll] for (uint n = 0; n < TN; n++) {
                            uint bc = localCol + n * TILE_SIZE;
                            acc[m][n] = mad(a0, (%SCALAR_DTYPE_OUTPUT%)tileB[lastFull][k    ][bc], acc[m][n]);
                            acc[m][n] = mad(a1, (%SCALAR_DTYPE_OUTPUT%)tileB[lastFull][k + 1][bc], acc[m][n]);
                            acc[m][n] = mad(a2, (%SCALAR_DTYPE_OUTPUT%)tileB[lastFull][k + 2][bc], acc[m][n]);
                            acc[m][n] = mad(a3, (%SCALAR_DTYPE_OUTPUT%)tileB[lastFull][k + 3][bc], acc[m][n]);
                        }
                    }
                }
            }
        } else if (hasPartialK) {
            // Only a partial tile (K < TILE_SIZE)
            [unroll] for (uint m = 0; m < TM; m++)
                tileA[0][localRow + m * TILE_SIZE][localCol] =
                    loadA(blockRowStart + localRow + m * TILE_SIZE, localCol);
            [unroll] for (uint n = 0; n < TN; n++)
                tileB[0][localRow][localCol + n * TILE_SIZE] =
                    loadB(localRow, blockColStart + localCol + n * TILE_SIZE);

            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                [unroll] for (uint m = 0; m < TM; m++) {
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[0][localRow + m * TILE_SIZE][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[0][localRow + m * TILE_SIZE][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[0][localRow + m * TILE_SIZE][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[0][localRow + m * TILE_SIZE][k + 3];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        uint bc = localCol + n * TILE_SIZE;
                        acc[m][n] = mad(a0, (%SCALAR_DTYPE_OUTPUT%)tileB[0][k    ][bc], acc[m][n]);
                        acc[m][n] = mad(a1, (%SCALAR_DTYPE_OUTPUT%)tileB[0][k + 1][bc], acc[m][n]);
                        acc[m][n] = mad(a2, (%SCALAR_DTYPE_OUTPUT%)tileB[0][k + 2][bc], acc[m][n]);
                        acc[m][n] = mad(a3, (%SCALAR_DTYPE_OUTPUT%)tileB[0][k + 3][bc], acc[m][n]);
                    }
                }
            }
        }

        // Unchecked output write
        [unroll] for (uint m = 0; m < TM; m++) {
            [unroll] for (uint n = 0; n < TN; n++) {
                uint outRow = blockRowStart + localRow + m * TILE_SIZE;
                uint outCol = blockColStart + localCol + n * TILE_SIZE;
                dataC[outRow * pc.strideB + outCol] = acc[m][n];
            }
        }
    } else {
        // Edge workgroup: single-buffered with checked loads
        for (uint t = 0; t < numTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            [unroll] for (uint m = 0; m < TM; m++)
                tileA[0][localRow + m * TILE_SIZE][localCol] =
                    loadA(blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            [unroll] for (uint n = 0; n < TN; n++)
                tileB[0][localRow][localCol + n * TILE_SIZE] =
                    loadB(tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);

            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                [unroll] for (uint m = 0; m < TM; m++) {
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[0][localRow + m * TILE_SIZE][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[0][localRow + m * TILE_SIZE][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[0][localRow + m * TILE_SIZE][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[0][localRow + m * TILE_SIZE][k + 3];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        uint bc = localCol + n * TILE_SIZE;
                        acc[m][n] = mad(a0, (%SCALAR_DTYPE_OUTPUT%)tileB[0][k    ][bc], acc[m][n]);
                        acc[m][n] = mad(a1, (%SCALAR_DTYPE_OUTPUT%)tileB[0][k + 1][bc], acc[m][n]);
                        acc[m][n] = mad(a2, (%SCALAR_DTYPE_OUTPUT%)tileB[0][k + 2][bc], acc[m][n]);
                        acc[m][n] = mad(a3, (%SCALAR_DTYPE_OUTPUT%)tileB[0][k + 3][bc], acc[m][n]);
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Checked output write
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
}
