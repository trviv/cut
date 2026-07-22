// Native CUDA counterpart of MatMulGemv8M.shader (M=2..16, 8 cols/WG,
// wave-per-row).
//
// Wave-per-row layout: the block is (32, ROWS_PER_WG) = 8 warps. Each warp
// computes the full-K dot product for its own row m = blockIdx.y*ROWS_PER_WG
// + threadIdx.y over the block's 8 columns. All 8 warps read the SAME B
// elements (k = threadIdx.x, stride WG_SIZE), so B loads are served from
// L1/broadcast and the effective B traffic is ~one read per ceil(M/8) rows.
// A CUDA warp is exactly (32 lanes, one threadIdx.y), so the reduction is a
// warp-local butterfly shuffle — no shared memory, no __syncthreads.
#include "ComputeOpsShared.h"

#ifndef WG_SIZE
#define WG_SIZE 32
#endif
#ifndef COLS_PER_WG
#define COLS_PER_WG 8
#endif
#ifndef ROWS_PER_WG
#define ROWS_PER_WG 8
#endif

#include "MatMulCommon.cuh"

extern "C" __global__ void cut_main(const cut_a_vec* __restrict__ dataA, const cut_b_vec* __restrict__ dataB, const cut_d_vec* __restrict__ dataD, cut_c_t* __restrict__ dataC, PushConstants pc) {
    uint lane = threadIdx.x;   // 0..31, lane within the warp
    uint wy = threadIdx.y;     // 0..ROWS_PER_WG-1, which row this warp owns
    uint baseN = blockIdx.x * COLS_PER_WG;
    uint m = blockIdx.y * ROWS_PER_WG + wy;

    if (baseN >= pc.N) return;

    bool rowValid = (m < pc.M);

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

    for (uint k = lane; k < pc.K; k += WG_SIZE) {
        cut_c_t a = rowValid ? (cut_c_t)mmLoadAFast(dataA, pc, m, k)
                             : (cut_c_t)(0);
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

    // Reduce the 32 lanes within this warp.
    #pragma unroll
    for (uint offset = 16; offset >= 1; offset >>= 1) {
        acc0 += WaveReadLaneAt(acc0, lane ^ offset);
        acc1 += WaveReadLaneAt(acc1, lane ^ offset);
        acc2 += WaveReadLaneAt(acc2, lane ^ offset);
        acc3 += WaveReadLaneAt(acc3, lane ^ offset);
        acc4 += WaveReadLaneAt(acc4, lane ^ offset);
        acc5 += WaveReadLaneAt(acc5, lane ^ offset);
        acc6 += WaveReadLaneAt(acc6, lane ^ offset);
        acc7 += WaveReadLaneAt(acc7, lane ^ offset);
    }

    if (rowValid && lane == 0) {
        uint col = baseN;
        if (col + 0 < pc.N) mmWriteOutput(dataC, dataD, pc, m, col + 0, acc0);
        if (col + 1 < pc.N) mmWriteOutput(dataC, dataD, pc, m, col + 1, acc1);
        if (col + 2 < pc.N) mmWriteOutput(dataC, dataD, pc, m, col + 2, acc2);
        if (col + 3 < pc.N) mmWriteOutput(dataC, dataD, pc, m, col + 3, acc3);
        if (col + 4 < pc.N) mmWriteOutput(dataC, dataD, pc, m, col + 4, acc4);
        if (col + 5 < pc.N) mmWriteOutput(dataC, dataD, pc, m, col + 5, acc5);
        if (col + 6 < pc.N) mmWriteOutput(dataC, dataD, pc, m, col + 6, acc6);
        if (col + 7 < pc.N) mmWriteOutput(dataC, dataD, pc, m, col + 7, acc7);
    }
}
