// Native CUDA counterpart of MatMulTiledRegVecBRegAligned.shader (aligned vec4 + B-register).
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
    uint tid = localRow * TILE_SIZE + localCol;

    uint2 tileId = mmSwizzle(pc, TILE_SIZE * TM, TILE_SIZE * TN, Gid.x, Gid.y);
    uint blockRowStart = tileId.y * TILE_SIZE * TM;
    uint blockColStart = tileId.x * TILE_SIZE * TN;

    cut_c_t acc[TM][TN];
    #pragma unroll
    for (uint m = 0; m < TM; m++)
        #pragma unroll
        for (uint n = 0; n < TN; n++)
            acc[m][n] = (cut_c_t)(0);

    uint numTiles = pc.K / TILE_SIZE;

    bool fullM = (blockRowStart + TILE_SIZE * TM <= pc.M);

    uint tileA_row = tid / (TILE_SIZE / 4);
    uint tileA_col = (tid % (TILE_SIZE / 4)) * 4;

    uint tileB_row = tid / (TILE_SIZE * TN / 4);
    uint tileB_col = (tid % (TILE_SIZE * TN / 4)) * 4;

    if (fullM) {
        for (uint t = 0; t < numTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            uint gA_idx = (blockRowStart + tileA_row) * pc.strideA + tileKStart + tileA_col;
            cut_a_vec vA = dataA[gA_idx >> 2];
            tileA[tileA_row][tileA_col + 0] = vA[0];
            tileA[tileA_row][tileA_col + 1] = vA[1];
            tileA[tileA_row][tileA_col + 2] = vA[2];
            tileA[tileA_row][tileA_col + 3] = vA[3];

            uint gB_idx = (tileKStart + tileB_row) * pc.strideB + blockColStart + tileB_col;
            cut_b_vec vB = dataB[gB_idx >> 2];
            tileB[tileB_row][tileB_col + 0] = vB[0];
            tileB[tileB_row][tileB_col + 1] = vB[1];
            tileB[tileB_row][tileB_col + 2] = vB[2];
            tileB[tileB_row][tileB_col + 3] = vB[3];

            __syncthreads();

            #pragma unroll
            for (uint k = 0; k < TILE_SIZE; k += 4) {
                cut_c_t b_reg[4][TN];
                #pragma unroll
                for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (cut_c_t)tileB[k    ][bc];
                    b_reg[1][n] = (cut_c_t)tileB[k + 1][bc];
                    b_reg[2][n] = (cut_c_t)tileB[k + 2][bc];
                    b_reg[3][n] = (cut_c_t)tileB[k + 3][bc];
                }
                #pragma unroll
                for (uint m = 0; m < TM; m++) {
                    cut_c_t a0 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k];
                    cut_c_t a1 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 1];
                    cut_c_t a2 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 2];
                    cut_c_t a3 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 3];
                    #pragma unroll
                    for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
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

            uint gA_row = blockRowStart + tileA_row;
            if (gA_row < pc.M) {
                uint gA_idx = gA_row * pc.strideA + tileKStart + tileA_col;
                cut_a_vec vA = dataA[gA_idx >> 2];
                tileA[tileA_row][tileA_col + 0] = vA[0];
                tileA[tileA_row][tileA_col + 1] = vA[1];
                tileA[tileA_row][tileA_col + 2] = vA[2];
                tileA[tileA_row][tileA_col + 3] = vA[3];
            } else {
                tileA[tileA_row][tileA_col + 0] = (cut_a_t)(0);
                tileA[tileA_row][tileA_col + 1] = (cut_a_t)(0);
                tileA[tileA_row][tileA_col + 2] = (cut_a_t)(0);
                tileA[tileA_row][tileA_col + 3] = (cut_a_t)(0);
            }

            uint gB_idx = (tileKStart + tileB_row) * pc.strideB + blockColStart + tileB_col;
            cut_b_vec vB = dataB[gB_idx >> 2];
            tileB[tileB_row][tileB_col + 0] = vB[0];
            tileB[tileB_row][tileB_col + 1] = vB[1];
            tileB[tileB_row][tileB_col + 2] = vB[2];
            tileB[tileB_row][tileB_col + 3] = vB[3];

            __syncthreads();

            #pragma unroll
            for (uint k = 0; k < TILE_SIZE; k += 4) {
                cut_c_t b_reg[4][TN];
                #pragma unroll
                for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (cut_c_t)tileB[k    ][bc];
                    b_reg[1][n] = (cut_c_t)tileB[k + 1][bc];
                    b_reg[2][n] = (cut_c_t)tileB[k + 2][bc];
                    b_reg[3][n] = (cut_c_t)tileB[k + 3][bc];
                }
                #pragma unroll
                for (uint m = 0; m < TM; m++) {
                    cut_c_t a0 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k];
                    cut_c_t a1 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 1];
                    cut_c_t a2 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 2];
                    cut_c_t a3 = (cut_c_t)tileA[localRow + m * TILE_SIZE][k + 3];
                    #pragma unroll
                    for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
                    }
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for (uint m = 0; m < TM; m++) {
            uint outRow = blockRowStart + localRow + m * TILE_SIZE;
            if (outRow < pc.M) {
                #pragma unroll
                for (uint n = 0; n < TN; n++) {
                    uint outCol = blockColStart + localCol + n * TILE_SIZE;
                    mmWriteOutput(dataC, dataD, pc, outRow, outCol, acc[m][n]);
                }
            }
        }
    }
}
