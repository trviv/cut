// Native CUDA counterpart of MatMulGemv8.shader (M=1 GEMV, 8 cols/WG, split-K).
//
// Split-K layout: the block is (32, WARPS_PER_WG) = 8 warps. All warps compute
// the SAME 8 output columns but each owns a different K stripe, so the grid is
// unchanged (ceil(N/8) blocks) while 8x more warps are resident. The previous
// one-warp-per-block version launched only ceil(N/8) warps total — for a decode
// matmul with N=576 that is 72 warps on an 82-SM GPU (0.9 warps/SM against a
// 48-64 warp capacity), so memory latency was never hidden and the kernel ran
// at ~12% of peak bandwidth. Each warp keeps 8 independent accumulators (ILP),
// reduces across its 32 lanes with a butterfly shuffle, publishes one partial
// per column to shared memory, and warp 0 sums the 8 partials.
#include "ComputeOpsShared.h"

#ifndef WG_SIZE
#define WG_SIZE 32
#endif
#ifndef COLS_PER_WG
#define COLS_PER_WG 8
#endif
#ifndef WARPS_PER_WG
#define WARPS_PER_WG 8
#endif

#include "MatMulCommon.cuh"

extern "C" __global__ void cut_main(const cut_a_vec* __restrict__ dataA, const cut_b_vec* __restrict__ dataB, const cut_d_vec* __restrict__ dataD, cut_c_t* __restrict__ dataC, PushConstants pc) {
    uint lane = threadIdx.x;   // 0..31, lane within the warp
    uint wy = threadIdx.y;     // 0..WARPS_PER_WG-1, which K stripe this warp owns
    uint baseN = blockIdx.x * COLS_PER_WG;
    uint m = blockIdx.y;

    // Block-uniform (depends only on blockIdx), so the __syncthreads() below is
    // safe: either every thread returns or none does.
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

    for (uint k = wy * WG_SIZE + lane; k < pc.K; k += WG_SIZE * WARPS_PER_WG) {
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

    // Reduce the 32 lanes within each warp.
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

    // One partial per (warp, column); warp 0 folds them into the final sums.
    __shared__ cut_c_t partial[WARPS_PER_WG][COLS_PER_WG];
    if (lane == 0) {
        partial[wy][0] = acc0;
        partial[wy][1] = acc1;
        partial[wy][2] = acc2;
        partial[wy][3] = acc3;
        partial[wy][4] = acc4;
        partial[wy][5] = acc5;
        partial[wy][6] = acc6;
        partial[wy][7] = acc7;
    }
    __syncthreads();

    if (wy == 0 && lane < COLS_PER_WG) {
        cut_c_t sum = (cut_c_t)(0);
        #pragma unroll
        for (uint w = 0; w < WARPS_PER_WG; ++w) {
            sum += partial[w][lane];
        }
        uint col = baseN + lane;
        if (col < pc.N) {
            mmWriteOutput(dataC, dataD, pc, m, col, sum);
        }
    }
}
