// MatMulQ8GemvKPar8.cu - Split-K matrix multiplication kernel for Q8 format
// This kernel uses 8 warps per block (32x8 threads), with each warp processing
// a different K stripe to improve memory latency hiding.

#include "MatMulQ8Common.cuh"

#define WG_SIZE 32
#define COLS_PER_WG 8
#define WARPS_PER_WG 8

extern "C" __global__ void cut_main(
    const CUT_VEC_DTYPE_INPUT1* __restrict__ dataA,
    const uint* __restrict__ packedB,
    const CUT_VEC_DTYPE_SCALES* __restrict__ scalesB,
    const CUT_VEC_DTYPE_OUTPUT* __restrict__ dataD,
    CUT_SCALAR_DTYPE_OUTPUT* __restrict__ dataC,
    PushConstants pc) {

    uint lane = threadIdx.x;   // 0..31 within the warp
    uint wy = threadIdx.y;     // 0..7, which K stripe this warp owns
    uint baseN = blockIdx.x * COLS_PER_WG;
    uint m = blockIdx.y;

    if (baseN >= pc.N || m >= pc.M) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    for (uint k = wy * WG_SIZE + lane; k < pc.K; k += WG_SIZE * WARPS_PER_WG) {
        float a = float(cut_loadA_fast(dataA, pc, m, k));
        uint byteIdx0 = k * pc.strideBN + baseN;
        uint packed0 = packedB[byteIdx0 >> 2];
        float b0_0, b1_0, b2_0, b3_0;
        cut_unpackB4(packed0, b0_0, b1_0, b2_0, b3_0);
        uint byteIdx1 = k * pc.strideBN + baseN + 4;
        uint packed1 = packedB[byteIdx1 >> 2];
        float b0_1, b1_1, b2_1, b3_1;
        cut_unpackB4(packed1, b0_1, b1_1, b2_1, b3_1);
        uint scaleBase0 = (k >> 5) * pc.scaleStride + baseN;
        float s0_0, s1_0, s2_0, s3_0;
        cut_loadScale4(scalesB, scaleBase0, s0_0, s1_0, s2_0, s3_0);
        uint scaleBase1 = (k >> 5) * pc.scaleStride + baseN + 4;
        float s0_1, s1_1, s2_1, s3_1;
        cut_loadScale4(scalesB, scaleBase1, s0_1, s1_1, s2_1, s3_1);
        acc0 = mad(a, b0_0 * s0_0, acc0);
        acc1 = mad(a, b1_0 * s1_0, acc1);
        acc2 = mad(a, b2_0 * s2_0, acc2);
        acc3 = mad(a, b3_0 * s3_0, acc3);
        acc4 = mad(a, b0_1 * s0_1, acc4);
        acc5 = mad(a, b1_1 * s1_1, acc5);
        acc6 = mad(a, b2_1 * s2_1, acc6);
        acc7 = mad(a, b3_1 * s3_1, acc7);
    }

    #pragma unroll
    for (uint offset = 16u; offset >= 1u; offset >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_xor_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_xor_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_xor_sync(0xffffffffu, acc3, offset);
        acc4 += __shfl_xor_sync(0xffffffffu, acc4, offset);
        acc5 += __shfl_xor_sync(0xffffffffu, acc5, offset);
        acc6 += __shfl_xor_sync(0xffffffffu, acc6, offset);
        acc7 += __shfl_xor_sync(0xffffffffu, acc7, offset);
    }

    __shared__ float partial[WARPS_PER_WG][COLS_PER_WG];
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
        float sum = 0.0f;
        #pragma unroll
        for (uint w = 0; w < WARPS_PER_WG; ++w) sum += partial[w][lane];
        uint col = baseN + lane;
        if (col < pc.N) cut_writeOutput(dataC, dataD, pc, m, col, sum);
    }
}
