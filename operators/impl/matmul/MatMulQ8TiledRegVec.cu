// MatMulQ8TiledRegVec.cu
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 4
#endif
#include "MatMulQ8Common.cuh"

extern "C" __global__ void cut_main(const CUT_VEC_DTYPE_INPUT1* __restrict__ dataA,
    const uint* __restrict__ packedB, const CUT_VEC_DTYPE_SCALES* __restrict__ scalesB,
    const CUT_VEC_DTYPE_OUTPUT* __restrict__ dataD, CUT_SCALAR_DTYPE_OUTPUT* __restrict__ dataC,
    PushConstants pc) {
    uint3 GTid;
    GTid.x = threadIdx.x;
    GTid.y = threadIdx.y;
    GTid.z = threadIdx.z;
    uint3 Gid;
    Gid.x = blockIdx.x;
    Gid.y = blockIdx.y;
    Gid.z = blockIdx.z;
    uint localRow = GTid.y;
    uint localCol = GTid.x;
    uint tid = localRow * TILE_SIZE + localCol;

    uint2 tileId = cut_swizzleTileId(pc, Gid);
    uint blockRowStart = tileId.y * TILE_SIZE * TM;
    uint blockColStart = tileId.x * TILE_SIZE * TN;

    CUT_SCALAR_DTYPE_OUTPUT acc[TM][TN];
    #pragma unroll
    for (uint m = 0; m < TM; m++)
        #pragma unroll
        for (uint n = 0; n < TN; n++)
            acc[m][n] = (CUT_SCALAR_DTYPE_OUTPUT)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;
    uint fullKTiles = pc.K / TILE_SIZE;
    bool hasPartialK = (pc.K % TILE_SIZE) != 0;
    bool fullM = (blockRowStart + TILE_SIZE * TM <= pc.M);
    bool fullN = (blockColStart + TILE_SIZE * TN <= pc.N);

    __shared__ CUT_SCALAR_DTYPE_INPUT1 tileA[TILE_SIZE * TM][TILE_SIZE + 1];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE * TN + 1];

    uint tileA_row = tid / (TILE_SIZE / 4);
    uint tileA_col = (tid % (TILE_SIZE / 4)) * 4;

    uint tileB_row = tid / (TILE_SIZE * TN / 4);
    uint tileB_col = (tid % (TILE_SIZE * TN / 4)) * 4;

    if (fullM && fullN) {
        for (uint t = 0; t < fullKTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            uint gA_idx = (blockRowStart + tileA_row) * pc.strideA + tileKStart + tileA_col;
            CUT_VEC_DTYPE_INPUT1 vA = dataA[gA_idx >> 2];
            tileA[tileA_row][tileA_col + 0] = vA[0];
            tileA[tileA_row][tileA_col + 1] = vA[1];
            tileA[tileA_row][tileA_col + 2] = vA[2];
            tileA[tileA_row][tileA_col + 3] = vA[3];

            {
                uint gB_k = tileKStart + tileB_row;
                uint gB_n = blockColStart + tileB_col;
                uint byteIdx = gB_k * pc.strideBN + gB_n;
                uint packed = packedB[byteIdx >> 2];

                float b0, b1, b2, b3;
                cut_unpackB4(packed, b0, b1, b2, b3);

                uint scaleBase = (gB_k >> 5) * pc.scaleStride + gB_n;
                float s0, s1, s2, s3;
                cut_loadScale4(scalesB, scaleBase, s0, s1, s2, s3);

                tileB[tileB_row][tileB_col + 0] = b0 * s0;
                tileB[tileB_row][tileB_col + 1] = b1 * s1;
                tileB[tileB_row][tileB_col + 2] = b2 * s2;
                tileB[tileB_row][tileB_col + 3] = b3 * s3;
            }

            __syncthreads();

            #pragma unroll
            for (uint k = 0; k < TILE_SIZE; k += 4) {
                CUT_SCALAR_DTYPE_OUTPUT b_reg[4][TN];
                #pragma unroll
                for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k    ][bc];
                    b_reg[1][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k + 1][bc];
                    b_reg[2][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k + 2][bc];
                    b_reg[3][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k + 3][bc];
                }
                #pragma unroll
                for (uint m = 0; m < TM; m++) {
                    CUT_SCALAR_DTYPE_OUTPUT a0 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k];
                    CUT_SCALAR_DTYPE_OUTPUT a1 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k + 1];
                    CUT_SCALAR_DTYPE_OUTPUT a2 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k + 2];
                    CUT_SCALAR_DTYPE_OUTPUT a3 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k + 3];
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

        if (hasPartialK) {
            uint tileKStart = fullKTiles * TILE_SIZE;

            #pragma unroll
            for (uint m = 0; m < TM; m++) {
                tileA[localRow + m * TILE_SIZE][localCol] =
                    cut_loadA(dataA, pc, blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            }
            #pragma unroll
            for (uint n = 0; n < TN; n++) {
                tileB[localRow][localCol + n * TILE_SIZE] =
                    cut_loadB(packedB, scalesB, pc, tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }

            __syncthreads();

            #pragma unroll
            for (uint k = 0; k < TILE_SIZE; k += 4) {
                CUT_SCALAR_DTYPE_OUTPUT b_reg[4][TN];
                #pragma unroll
                for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k    ][bc];
                    b_reg[1][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k + 1][bc];
                    b_reg[2][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k + 2][bc];
                    b_reg[3][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k + 3][bc];
                }
                #pragma unroll
                for (uint m = 0; m < TM; m++) {
                    CUT_SCALAR_DTYPE_OUTPUT a0 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k];
                    CUT_SCALAR_DTYPE_OUTPUT a1 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k + 1];
                    CUT_SCALAR_DTYPE_OUTPUT a2 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k + 2];
                    CUT_SCALAR_DTYPE_OUTPUT a3 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k + 3];
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
                cut_writeOutput(dataC, dataD, pc, outRow, outCol, acc[m][n]);
            }
        }
    } else {
        for (uint t = 0; t < numTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            #pragma unroll
            for (uint m = 0; m < TM; m++) {
                tileA[localRow + m * TILE_SIZE][localCol] =
                    cut_loadA(dataA, pc, blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            }
            #pragma unroll
            for (uint n = 0; n < TN; n++) {
                tileB[localRow][localCol + n * TILE_SIZE] =
                    cut_loadB(packedB, scalesB, pc, tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }

            __syncthreads();

            #pragma unroll
            for (uint k = 0; k < TILE_SIZE; k += 4) {
                CUT_SCALAR_DTYPE_OUTPUT b_reg[4][TN];
                #pragma unroll
                for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k    ][bc];
                    b_reg[1][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k + 1][bc];
                    b_reg[2][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k + 2][bc];
                    b_reg[3][n] = (CUT_SCALAR_DTYPE_OUTPUT)tileB[k + 3][bc];
                }
                #pragma unroll
                for (uint m = 0; m < TM; m++) {
                    CUT_SCALAR_DTYPE_OUTPUT a0 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k];
                    CUT_SCALAR_DTYPE_OUTPUT a1 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k + 1];
                    CUT_SCALAR_DTYPE_OUTPUT a2 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k + 2];
                    CUT_SCALAR_DTYPE_OUTPUT a3 = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k + 3];
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
                if (outRow < pc.M && outCol < pc.N) {
                    cut_writeOutput(dataC, dataD, pc, outRow, outCol, acc[m][n]);
                }
            }
        }
    }
}
