// Native CUDA counterpart of MatMulGemv.shader (M=1 GEMV, K-parallel warp reduction).
#include "ComputeOpsShared.h"

#ifndef WG_SIZE
#define WG_SIZE 32
#endif
#ifndef COLS_PER_WG
#define COLS_PER_WG 4
#endif

#include "MatMulCommon.cuh"

extern "C" __global__ void cut_main(const cut_a_vec* __restrict__ dataA, const cut_b_vec* __restrict__ dataB, const cut_d_vec* __restrict__ dataD, cut_c_t* __restrict__ dataC, PushConstants pc) {
    uint tid = threadIdx.x;
    uint baseN = blockIdx.x * COLS_PER_WG;
    uint m = blockIdx.y;

    if (baseN >= pc.N || m >= pc.M) return;

    cut_c_t acc0 = (cut_c_t)(0);
    cut_c_t acc1 = (cut_c_t)(0);
    cut_c_t acc2 = (cut_c_t)(0);
    cut_c_t acc3 = (cut_c_t)(0);

    uint vec4Col = baseN >> 2;
    uint strideB4 = pc.strideB >> 2;

    for (uint k = tid; k < pc.K; k += WG_SIZE) {
        cut_a_t a = (cut_a_t)mmLoadAFast(dataA, pc, m, k);
        cut_b_vec bVec = dataB[k * strideB4 + vec4Col];
        acc0 = mad(a, (cut_c_t)bVec[0], acc0);
        acc1 = mad(a, (cut_c_t)bVec[1], acc1);
        acc2 = mad(a, (cut_c_t)bVec[2], acc2);
        acc3 = mad(a, (cut_c_t)bVec[3], acc3);
    }

#pragma unroll
    for (uint offset = 16; offset >= 1; offset >>= 1) {
        acc0 += WaveReadLaneAt(acc0, tid ^ offset);
        acc1 += WaveReadLaneAt(acc1, tid ^ offset);
        acc2 += WaveReadLaneAt(acc2, tid ^ offset);
        acc3 += WaveReadLaneAt(acc3, tid ^ offset);
    }

    if (tid == 0) {
        uint colCount = min(COLS_PER_WG, pc.N - baseN);
        mmWriteOutput(dataC, dataD, pc, m, baseN, acc0);
        if (colCount > 1) mmWriteOutput(dataC, dataD, pc, m, baseN + 1, acc1);
        if (colCount > 2) mmWriteOutput(dataC, dataD, pc, m, baseN + 2, acc2);
        if (colCount > 3) mmWriteOutput(dataC, dataD, pc, m, baseN + 3, acc3);
    }
}
