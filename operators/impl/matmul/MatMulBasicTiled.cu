// Native CUDA counterpart of MatMulBasicTiled.shader (shared-memory 16x16 tiled matmul).
#include "ComputeOpsShared.h"

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#include "MatMulCommon.cuh"

extern "C" __global__ void cut_main(const cut_a_vec* __restrict__ dataA, const cut_b_vec* __restrict__ dataB, const cut_d_vec* __restrict__ dataD, cut_c_t* __restrict__ dataC, PushConstants pc) {
    __shared__ cut_a_t tileA[TILE_SIZE][TILE_SIZE];
    __shared__ cut_b_t tileB[TILE_SIZE][TILE_SIZE];

    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    uint localRow = threadIdx.y;
    uint localCol = threadIdx.x;

    cut_c_t sum = (cut_c_t)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + localCol;
        tileA[localRow][localCol] = mmLoadA(dataA, pc, row, aCol);

        uint bRow = t * TILE_SIZE + localRow;
        tileB[localRow][localCol] = mmLoadB(dataB, pc, bRow, col);

        __syncthreads();

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += (cut_c_t)(tileA[localRow][k]) * (cut_c_t)(tileB[k][localCol]);
        }

        __syncthreads();
    }

    if (row < pc.M && col < pc.N) {
        mmWriteOutput(dataC, dataD, pc, row, col, sum);
    }
}
