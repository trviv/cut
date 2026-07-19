// Native CUDA counterpart of MatMulTiledReg.shader (tiled register-blocked matmul).
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
    __shared__ cut_a_t tileA[TILE_SIZE * TM][TILE_SIZE + 1];
    __shared__ cut_b_t tileB[TILE_SIZE][TILE_SIZE * TN + 1];

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
        for (uint t = 0; t < fullKTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            #pragma unroll
            for (uint m = 0; m < TM; m++) {
                tileA[localRow + m * TILE_SIZE][localCol] =
                    mmLoadAFast(dataA, pc, blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            }
            #pragma unroll
            for (uint n = 0; n < TN; n++) {
                tileB[localRow][localCol + n * TILE_SIZE] =
                    mmLoadBFast(dataB, pc, tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }

            __syncthreads();

            #pragma unroll
            for (uint k = 0; k < TILE_SIZE; k += 4) {
                #pragma unroll
                for (uint m = 0; m < TM; m++) {
                    cut_c_t a0 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k];
                    cut_c_t a1 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 1];
                    cut_c_t a2 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 2];
                    cut_c_t a3 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 3];
                    #pragma unroll
                    for (uint n = 0; n < TN; n++) {
                        uint bc = localCol + n * TILE_SIZE;
                        acc[m][n] = mad(a0, (cut_c_t)tileB[k][bc], acc[m][n]);
                        acc[m][n] = mad(a1, (cut_c_t)tileB[k + 1][bc], acc[m][n]);
                        acc[m][n] = mad(a2, (cut_c_t)tileB[k + 2][bc], acc[m][n]);
                        acc[m][n] = mad(a3, (cut_c_t)tileB[k + 3][bc], acc[m][n]);
                    }
                }
            }

            __syncthreads();
        }

        if (hasPartialK) {
            uint tileKStart = fullKTiles * TILE_SIZE;

            #pragma unroll
            for (uint m = 0; m < TM; m++) {
                tileA[localRow + m * TILE_SIZE][localCol] =
                    mmLoadA(dataA, pc, blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            }
            #pragma unroll
            for (uint n = 0; n < TN; n++) {
                tileB[localRow][localCol + n * TILE_SIZE] =
                    mmLoadB(dataB, pc, tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }

            __syncthreads();

            #pragma unroll
            for (uint k = 0; k < TILE_SIZE; k += 4) {
                #pragma unroll
                for (uint m = 0; m < TM; m++) {
                    cut_c_t a0 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k];
                    cut_c_t a1 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 1];
                    cut_c_t a2 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 2];
                    cut_c_t a3 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 3];
                    #pragma unroll
                    for (uint n = 0; n < TN; n++) {
                        uint bc = localCol + n * TILE_SIZE;
                        acc[m][n] = mad(a0, (cut_c_t)tileB[k][bc], acc[m][n]);
                        acc[m][n] = mad(a1, (cut_c_t)tileB[k + 1][bc], acc[m][n]);
                        acc[m][n] = mad(a2, (cut_c_t)tileB[k + 2][bc], acc[m][n]);
                        acc[m][n] = mad(a3, (cut_c_t)tileB[k + 3][bc], acc[m][n]);
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
                mmWriteOutput(dataC, dataD, pc, outRow, outCol, acc[m][n]);
            }
        }
    } else {
        for (uint t = 0; t < numTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            #pragma unroll
            for (uint m = 0; m < TM; m++) {
                tileA[localRow + m * TILE_SIZE][localCol] =
                    mmLoadA(dataA, pc, blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            }
            #pragma unroll
            for (uint n = 0; n < TN; n++) {
                tileB[localRow][localCol + n * TILE_SIZE] =
                    mmLoadB(dataB, pc, tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }

            __syncthreads();

            #pragma unroll
            for (uint k = 0; k < TILE_SIZE; k += 4) {
                #pragma unroll
                for (uint m = 0; m < TM; m++) {
                    cut_c_t a0 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k];
                    cut_c_t a1 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 1];
                    cut_c_t a2 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 2];
                    cut_c_t a3 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 3];
                    #pragma unroll
                    for (uint n = 0; n < TN; n++) {
                        uint bc = localCol + n * TILE_SIZE;
                        acc[m][n] = mad(a0, (cut_c_t)tileB[k][bc], acc[m][n]);
                        acc[m][n] = mad(a1, (cut_c_t)tileB[k + 1][bc], acc[m][n]);
                        acc[m][n] = mad(a2, (cut_c_t)tileB[k + 2][bc], acc[m][n]);
                        acc[m][n] = mad(a3, (cut_c_t)tileB[k + 3][bc], acc[m][n]);
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
