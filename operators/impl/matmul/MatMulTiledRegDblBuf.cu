// Native CUDA counterpart of MatMulTiledRegDblBuf.shader (double-buffered shared tiles).
#include "ComputeOpsShared.h"

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 4
#endif

#include "MatMulCommon.cuh"

extern "C" __global__ void cut_main(const cut_a_vec* __restrict__ dataA, const cut_b_vec* __restrict__ dataB, const cut_d_vec* __restrict__ dataD, cut_c_t* __restrict__ dataC, PushConstants pc) {
    __shared__ cut_a_t tileA[2][TILE_SIZE * TM][TILE_SIZE + 1];
    __shared__ cut_b_t tileB[2][TILE_SIZE][TILE_SIZE * TN + 1];

    uint3 GTid;
    GTid.x = threadIdx.x;
    GTid.y = threadIdx.y;
    uint3 Gid;
    Gid.x = blockIdx.x;
    Gid.y = blockIdx.y;

    uint localRow = GTid.y;
    uint localCol = GTid.x;

    uint2 tileId = mmSwizzle(pc, TILE_SIZE * TM, TILE_SIZE * TN, Gid.x, Gid.y);
    uint blockRowStart = tileId.y * TILE_SIZE * TM;
    uint blockColStart = tileId.x * TILE_SIZE * TN;

    cut_c_t acc[TM][TN];
    #pragma unroll
    for (uint m = 0; m < TM; m++)
        #pragma unroll
        for (uint n = 0; n < TN; n++)
            acc[m][n] = (cut_c_t)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;
    uint fullKTiles = pc.K / TILE_SIZE;
    bool hasPartialK = (pc.K % TILE_SIZE) != 0;
    bool fullM = (blockRowStart + TILE_SIZE * TM <= pc.M);
    bool fullN = (blockColStart + TILE_SIZE * TN <= pc.N);

    if (fullM && fullN) {
        if (fullKTiles > 0) {
            {
                uint tileKStart = 0;
                #pragma unroll
                for (uint m = 0; m < TM; m++)
                    tileA[0][localRow + m * TILE_SIZE][localCol] =
                        mmLoadAFast(dataA, pc, blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
                #pragma unroll
                for (uint n = 0; n < TN; n++)
                    tileB[0][localRow][localCol + n * TILE_SIZE] =
                        mmLoadBFast(dataB, pc, tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }
            __syncthreads();

            for (uint t = 1; t < fullKTiles; t++) {
                uint cur = (t - 1) & 1;
                uint nxt = t & 1;
                uint tileKStart = t * TILE_SIZE;

                #pragma unroll
                for (uint m = 0; m < TM; m++)
                    tileA[nxt][localRow + m * TILE_SIZE][localCol] =
                        mmLoadAFast(dataA, pc, blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
                #pragma unroll
                for (uint n = 0; n < TN; n++)
                    tileB[nxt][localRow][localCol + n * TILE_SIZE] =
                        mmLoadBFast(dataB, pc, tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);

                #pragma unroll
                for (uint k = 0; k < TILE_SIZE; k += 4) {
                    #pragma unroll
                    for (uint m = 0; m < TM; m++) {
                        cut_c_t a0 = (cut_c_t)tileA[cur][localRow + m * TILE_SIZE][k];
                        cut_c_t a1 = (cut_c_t)tileA[cur][localRow + m * TILE_SIZE][k + 1];
                        cut_c_t a2 = (cut_c_t)tileA[cur][localRow + m * TILE_SIZE][k + 2];
                        cut_c_t a3 = (cut_c_t)tileA[cur][localRow + m * TILE_SIZE][k + 3];
                        #pragma unroll
                        for (uint n = 0; n < TN; n++) {
                            uint bc = localCol + n * TILE_SIZE;
                            acc[m][n] = mad(a0, (cut_c_t)tileB[cur][k][bc], acc[m][n]);
                            acc[m][n] = mad(a1, (cut_c_t)tileB[cur][k + 1][bc], acc[m][n]);
                            acc[m][n] = mad(a2, (cut_c_t)tileB[cur][k + 2][bc], acc[m][n]);
                            acc[m][n] = mad(a3, (cut_c_t)tileB[cur][k + 3][bc], acc[m][n]);
                        }
                    }
                }

                __syncthreads();
            }

            if (hasPartialK) {
                uint lastFull = (fullKTiles - 1) & 1;
                uint partBuf = fullKTiles & 1;
                uint partKStart = fullKTiles * TILE_SIZE;

                #pragma unroll
                for (uint m = 0; m < TM; m++)
                    tileA[partBuf][localRow + m * TILE_SIZE][localCol] =
                        mmLoadA(dataA, pc, blockRowStart + localRow + m * TILE_SIZE, partKStart + localCol);
                #pragma unroll
                for (uint n = 0; n < TN; n++)
                    tileB[partBuf][localRow][localCol + n * TILE_SIZE] =
                        mmLoadB(dataB, pc, partKStart + localRow, blockColStart + localCol + n * TILE_SIZE);

                #pragma unroll
                for (uint k = 0; k < TILE_SIZE; k += 4) {
                    #pragma unroll
                    for (uint m = 0; m < TM; m++) {
                        cut_c_t a0 = (cut_c_t)tileA[lastFull][localRow + m * TILE_SIZE][k];
                        cut_c_t a1 = (cut_c_t)tileA[lastFull][localRow + m * TILE_SIZE][k + 1];
                        cut_c_t a2 = (cut_c_t)tileA[lastFull][localRow + m * TILE_SIZE][k + 2];
                        cut_c_t a3 = (cut_c_t)tileA[lastFull][localRow + m * TILE_SIZE][k + 3];
                        #pragma unroll
                        for (uint n = 0; n < TN; n++) {
                            uint bc = localCol + n * TILE_SIZE;
                            acc[m][n] = mad(a0, (cut_c_t)tileB[lastFull][k][bc], acc[m][n]);
                            acc[m][n] = mad(a1, (cut_c_t)tileB[lastFull][k + 1][bc], acc[m][n]);
                            acc[m][n] = mad(a2, (cut_c_t)tileB[lastFull][k + 2][bc], acc[m][n]);
                            acc[m][n] = mad(a3, (cut_c_t)tileB[lastFull][k + 3][bc], acc[m][n]);
                        }
                    }
                }

                __syncthreads();

                #pragma unroll
                for (uint k = 0; k < TILE_SIZE; k += 4) {
                    #pragma unroll
                    for (uint m = 0; m < TM; m++) {
                        cut_c_t a0 = (cut_c_t)tileA[partBuf][localRow + m * TILE_SIZE][k];
                        cut_c_t a1 = (cut_c_t)tileA[partBuf][localRow + m * TILE_SIZE][k + 1];
                        cut_c_t a2 = (cut_c_t)tileA[partBuf][localRow + m * TILE_SIZE][k + 2];
                        cut_c_t a3 = (cut_c_t)tileA[partBuf][localRow + m * TILE_SIZE][k + 3];
                        #pragma unroll
                        for (uint n = 0; n < TN; n++) {
                            uint bc = localCol + n * TILE_SIZE;
                            acc[m][n] = mad(a0, (cut_c_t)tileB[partBuf][k][bc], acc[m][n]);
                            acc[m][n] = mad(a1, (cut_c_t)tileB[partBuf][k + 1][bc], acc[m][n]);
                            acc[m][n] = mad(a2, (cut_c_t)tileB[partBuf][k + 2][bc], acc[m][n]);
                            acc[m][n] = mad(a3, (cut_c_t)tileB[partBuf][k + 3][bc], acc[m][n]);
                        }
                    }
                }
            } else {
                uint lastFull = (fullKTiles - 1) & 1;
                #pragma unroll
                for (uint k = 0; k < TILE_SIZE; k += 4) {
                    #pragma unroll
                    for (uint m = 0; m < TM; m++) {
                        cut_c_t a0 = (cut_c_t)tileA[lastFull][localRow + m * TILE_SIZE][k];
                        cut_c_t a1 = (cut_c_t)tileA[lastFull][localRow + m * TILE_SIZE][k + 1];
                        cut_c_t a2 = (cut_c_t)tileA[lastFull][localRow + m * TILE_SIZE][k + 2];
                        cut_c_t a3 = (cut_c_t)tileA[lastFull][localRow + m * TILE_SIZE][k + 3];
                        #pragma unroll
                        for (uint n = 0; n < TN; n++) {
                            uint bc = localCol + n * TILE_SIZE;
                            acc[m][n] = mad(a0, (cut_c_t)tileB[lastFull][k][bc], acc[m][n]);
                            acc[m][n] = mad(a1, (cut_c_t)tileB[lastFull][k + 1][bc], acc[m][n]);
                            acc[m][n] = mad(a2, (cut_c_t)tileB[lastFull][k + 2][bc], acc[m][n]);
                            acc[m][n] = mad(a3, (cut_c_t)tileB[lastFull][k + 3][bc], acc[m][n]);
                        }
                    }
                }
            }
        } else if (hasPartialK) {
            #pragma unroll
            for (uint m = 0; m < TM; m++)
                tileA[0][localRow + m * TILE_SIZE][localCol] =
                    mmLoadA(dataA, pc, blockRowStart + localRow + m * TILE_SIZE, localCol);
            #pragma unroll
            for (uint n = 0; n < TN; n++)
                tileB[0][localRow][localCol + n * TILE_SIZE] =
                    mmLoadB(dataB, pc, localRow, blockColStart + localCol + n * TILE_SIZE);

            __syncthreads();

            #pragma unroll
            for (uint k = 0; k < TILE_SIZE; k += 4) {
                #pragma unroll
                for (uint m = 0; m < TM; m++) {
                    cut_c_t a0 = (cut_c_t)tileA[0][localRow + m * TILE_SIZE][k];
                    cut_c_t a1 = (cut_c_t)tileA[0][localRow + m * TILE_SIZE][k + 1];
                    cut_c_t a2 = (cut_c_t)tileA[0][localRow + m * TILE_SIZE][k + 2];
                    cut_c_t a3 = (cut_c_t)tileA[0][localRow + m * TILE_SIZE][k + 3];
                    #pragma unroll
                    for (uint n = 0; n < TN; n++) {
                        uint bc = localCol + n * TILE_SIZE;
                        acc[m][n] = mad(a0, (cut_c_t)tileB[0][k][bc], acc[m][n]);
                        acc[m][n] = mad(a1, (cut_c_t)tileB[0][k + 1][bc], acc[m][n]);
                        acc[m][n] = mad(a2, (cut_c_t)tileB[0][k + 2][bc], acc[m][n]);
                        acc[m][n] = mad(a3, (cut_c_t)tileB[0][k + 3][bc], acc[m][n]);
                    }
                }
            }
        }

        #pragma unroll
        for (uint m = 0; m < TM; m++) {
            #pragma unroll
            for (uint n = 0; n < TN; n++) {
                uint outRow = blockRowStart + localRow + m * TILE_SIZE;
                uint outCol = blockColStart + localCol + n * TILE_SIZE;
                mmWriteOutput(dataC, dataD, pc, outRow, outCol, acc[m][n]);
            }
        }
    } else {
        for (uint t = 0; t < numTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            #pragma unroll
            for (uint m = 0; m < TM; m++)
                tileA[0][localRow + m * TILE_SIZE][localCol] =
                    mmLoadA(dataA, pc, blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            #pragma unroll
            for (uint n = 0; n < TN; n++)
                tileB[0][localRow][localCol + n * TILE_SIZE] =
                    mmLoadB(dataB, pc, tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);

            __syncthreads();

            #pragma unroll
            for (uint k = 0; k < TILE_SIZE; k += 4) {
                #pragma unroll
                for (uint m = 0; m < TM; m++) {
                    cut_c_t a0 = (cut_c_t)tileA[0][localRow + m * TILE_SIZE][k];
                    cut_c_t a1 = (cut_c_t)tileA[0][localRow + m * TILE_SIZE][k + 1];
                    cut_c_t a2 = (cut_c_t)tileA[0][localRow + m * TILE_SIZE][k + 2];
                    cut_c_t a3 = (cut_c_t)tileA[0][localRow + m * TILE_SIZE][k + 3];
                    #pragma unroll
                    for (uint n = 0; n < TN; n++) {
                        uint bc = localCol + n * TILE_SIZE;
                        acc[m][n] = mad(a0, (cut_c_t)tileB[0][k][bc], acc[m][n]);
                        acc[m][n] = mad(a1, (cut_c_t)tileB[0][k + 1][bc], acc[m][n]);
                        acc[m][n] = mad(a2, (cut_c_t)tileB[0][k + 2][bc], acc[m][n]);
                        acc[m][n] = mad(a3, (cut_c_t)tileB[0][k + 3][bc], acc[m][n]);
                    }
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for (uint m = 0; m < TM; m++) {
            #pragma unroll
            for (uint n = 0; n < TN; n++) {
                uint outRow = blockRowStart + localRow + m * TILE_SIZE;
                uint outCol = blockColStart + localCol + n * TILE_SIZE;
                if (outRow < pc.M && outCol < pc.N) {
                    mmWriteOutput(dataC, dataD, pc, outRow, outCol, acc[m][n]);
                }
            }
        }
    }
}
