// MatMulQ4TiledReg.cu
// Native CUDA counterpart of MatMulQ4TiledReg.shader (Q4_0 dequant register-tiled matmul).
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 4
#endif
#include "MatMulQ4Common.cuh"

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

    __shared__ CUT_SCALAR_DTYPE_INPUT1 tileA[TILE_SIZE * TM][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE * TN];

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

    for (uint t = 0; t < numTiles; t++) {
        uint tileKStart = t * TILE_SIZE;

        #pragma unroll
        for (uint m = 0; m < TM; m++) {
            uint aRow = blockRowStart + localRow + m * TILE_SIZE;
            uint aCol = tileKStart + localCol;
            tileA[localRow + m * TILE_SIZE][localCol] = cut_loadA(dataA, pc, aRow, aCol);
        }

        #pragma unroll
        for (uint n = 0; n < TN; n++) {
            uint bK = tileKStart + localRow;
            uint bN = blockColStart + localCol + n * TILE_SIZE;
            tileB[localRow][localCol + n * TILE_SIZE] = cut_loadB(packedB, scalesB, pc, bK, bN);
        }

        __syncthreads();

        for (uint k = 0; k < TILE_SIZE; k++) {
            #pragma unroll
            for (uint m = 0; m < TM; m++) {
                CUT_SCALAR_DTYPE_OUTPUT aVal = (CUT_SCALAR_DTYPE_OUTPUT)tileA[localRow + m * TILE_SIZE][k];
                #pragma unroll
                for (uint n = 0; n < TN; n++) {
                    acc[m][n] += aVal * (CUT_SCALAR_DTYPE_OUTPUT)tileB[k][localCol + n * TILE_SIZE];
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
