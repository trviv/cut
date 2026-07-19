// Native CUDA counterpart of MatMulGemv8.shader (M=1 GEMV, 8 cols/WG, K-unroll x4).
#include "ComputeOpsShared.h"

#ifndef WG_SIZE
#define WG_SIZE 32
#endif
#ifndef COLS_PER_WG
#define COLS_PER_WG 8
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
    cut_c_t acc4 = (cut_c_t)(0);
    cut_c_t acc5 = (cut_c_t)(0);
    cut_c_t acc6 = (cut_c_t)(0);
    cut_c_t acc7 = (cut_c_t)(0);

    uint vec4Col0 = baseN >> 2;
    uint vec4Col1 = (baseN + 4) >> 2;
    uint strideB4 = pc.strideB >> 2;

    uint K4 = pc.K & ~(4u * WG_SIZE - 1u);
    uint k = tid;

    for (; k < K4; k += 4 * WG_SIZE) {
        {
            cut_c_t a = (cut_c_t)mmLoadAFast(dataA, pc, m, k);
            cut_b_vec bVec0 = dataB[k * strideB4 + vec4Col0];
            cut_b_vec bVec1 = dataB[k * strideB4 + vec4Col1];
            acc0 = mad(a, (cut_c_t)bVec0[0], acc0);
            acc1 = mad(a, (cut_c_t)bVec0[1], acc1);
            acc2 = mad(a, (cut_c_t)bVec0[2], acc2);
            acc3 = mad(a, (cut_c_t)bVec0[3], acc3);
            acc4 = mad(a, (cut_c_t)bVec1[0], acc4);
            acc5 = mad(a, (cut_c_t)bVec1[1], acc5);
            acc6 = mad(a, (cut_c_t)bVec1[2], acc6);
            acc7 = mad(a, (cut_c_t)bVec1[3], acc7);
        }
        {
            uint k1 = k + WG_SIZE;
            cut_c_t a = (cut_c_t)mmLoadAFast(dataA, pc, m, k1);
            cut_b_vec bVec0 = dataB[k1 * strideB4 + vec4Col0];
            cut_b_vec bVec1 = dataB[k1 * strideB4 + vec4Col1];
            acc0 = mad(a, (cut_c_t)bVec0[0], acc0);
            acc1 = mad(a, (cut_c_t)bVec0[1], acc1);
            acc2 = mad(a, (cut_c_t)bVec0[2], acc2);
            acc3 = mad(a, (cut_c_t)bVec0[3], acc3);
            acc4 = mad(a, (cut_c_t)bVec1[0], acc4);
            acc5 = mad(a, (cut_c_t)bVec1[1], acc5);
            acc6 = mad(a, (cut_c_t)bVec1[2], acc6);
            acc7 = mad(a, (cut_c_t)bVec1[3], acc7);
        }
        {
            uint k2 = k + 2 * WG_SIZE;
            cut_c_t a = (cut_c_t)mmLoadAFast(dataA, pc, m, k2);
            cut_b_vec bVec0 = dataB[k2 * strideB4 + vec4Col0];
            cut_b_vec bVec1 = dataB[k2 * strideB4 + vec4Col1];
            acc0 = mad(a, (cut_c_t)bVec0[0], acc0);
            acc1 = mad(a, (cut_c_t)bVec0[1], acc1);
            acc2 = mad(a, (cut_c_t)bVec0[2], acc2);
            acc3 = mad(a, (cut_c_t)bVec0[3], acc3);
            acc4 = mad(a, (cut_c_t)bVec1[0], acc4);
            acc5 = mad(a, (cut_c_t)bVec1[1], acc5);
            acc6 = mad(a, (cut_c_t)bVec1[2], acc6);
            acc7 = mad(a, (cut_c_t)bVec1[3], acc7);
        }
        {
            uint k3 = k + 3 * WG_SIZE;
            cut_c_t a = (cut_c_t)mmLoadAFast(dataA, pc, m, k3);
            cut_b_vec bVec0 = dataB[k3 * strideB4 + vec4Col0];
            cut_b_vec bVec1 = dataB[k3 * strideB4 + vec4Col1];
            acc0 = mad(a, (cut_c_t)bVec0[0], acc0);
            acc1 = mad(a, (cut_c_t)bVec0[1], acc1);
            acc2 = mad(a, (cut_c_t)bVec0[2], acc2);
            acc3 = mad(a, (cut_c_t)bVec0[3], acc3);
            acc4 = mad(a, (cut_c_t)bVec1[0], acc4);
            acc5 = mad(a, (cut_c_t)bVec1[1], acc5);
            acc6 = mad(a, (cut_c_t)bVec1[2], acc6);
            acc7 = mad(a, (cut_c_t)bVec1[3], acc7);
        }
    }

    for (; k < pc.K; k += WG_SIZE) {
        cut_c_t a = (cut_c_t)mmLoadAFast(dataA, pc, m, k);
        cut_b_vec bVec0 = dataB[k * strideB4 + vec4Col0];
        cut_b_vec bVec1 = dataB[k * strideB4 + vec4Col1];
        acc0 = mad(a, (cut_c_t)bVec0[0], acc0);
        acc1 = mad(a, (cut_c_t)bVec0[1], acc1);
        acc2 = mad(a, (cut_c_t)bVec0[2], acc2);
        acc3 = mad(a, (cut_c_t)bVec0[3], acc3);
        acc4 = mad(a, (cut_c_t)bVec1[0], acc4);
        acc5 = mad(a, (cut_c_t)bVec1[1], acc5);
        acc6 = mad(a, (cut_c_t)bVec1[2], acc6);
        acc7 = mad(a, (cut_c_t)bVec1[3], acc7);
    }

    #pragma unroll
    for (uint offset = 16; offset >= 1; offset >>= 1) {
        acc0 += WaveReadLaneAt(acc0, tid ^ offset);
        acc1 += WaveReadLaneAt(acc1, tid ^ offset);
        acc2 += WaveReadLaneAt(acc2, tid ^ offset);
        acc3 += WaveReadLaneAt(acc3, tid ^ offset);
        acc4 += WaveReadLaneAt(acc4, tid ^ offset);
        acc5 += WaveReadLaneAt(acc5, tid ^ offset);
        acc6 += WaveReadLaneAt(acc6, tid ^ offset);
        acc7 += WaveReadLaneAt(acc7, tid ^ offset);
    }

    if (tid == 0) {
        uint colCount = min(COLS_PER_WG, pc.N - baseN);
        mmWriteOutput(dataC, dataD, pc, m, baseN, acc0);
        if (colCount > 1) mmWriteOutput(dataC, dataD, pc, m, baseN + 1, acc1);
        if (colCount > 2) mmWriteOutput(dataC, dataD, pc, m, baseN + 2, acc2);
        if (colCount > 3) mmWriteOutput(dataC, dataD, pc, m, baseN + 3, acc3);
        if (colCount > 4) mmWriteOutput(dataC, dataD, pc, m, baseN + 4, acc4);
        if (colCount > 5) mmWriteOutput(dataC, dataD, pc, m, baseN + 5, acc5);
        if (colCount > 6) mmWriteOutput(dataC, dataD, pc, m, baseN + 6, acc6);
        if (colCount > 7) mmWriteOutput(dataC, dataD, pc, m, baseN + 7, acc7);
    }
}
