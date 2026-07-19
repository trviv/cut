// Native CUDA counterpart of MatMulNaive.shader (naive matmul, no tiling).
#include "ComputeOpsShared.h"

#include "MatMulCommon.cuh"

extern "C" __global__ void cut_main(const cut_a_vec* __restrict__ dataA, const cut_b_vec* __restrict__ dataB, const cut_d_vec* __restrict__ dataD, cut_c_t* __restrict__ dataC, PushConstants pc) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= pc.M || col >= pc.N) return;

    cut_c_t sum = (cut_c_t)(0);
    for (uint k = 0; k < pc.K; k++) {
        uint idxA = row * pc.strideA + k;
        uint idxB = k * pc.strideB + col;
        sum += (cut_c_t)(dataA[idxA >> 2][idxA & 3]) * (cut_c_t)(dataB[idxB >> 2][idxB & 3]);
    }

    mmWriteOutput(dataC, dataD, pc, row, col, sum);
}
