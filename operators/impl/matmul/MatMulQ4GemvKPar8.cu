// MatMulQ4GemvKPar8.cu
#include "MatMulQ4Common.cuh"

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

    // Split-K: wave wy walks the stripe k = wy*32 + lane, stride 32*8.
    // The 8 independent accumulators supply the ILP the manual unroll gave.
    for (uint k = wy * WG_SIZE + lane; k < pc.K; k += WG_SIZE * WARPS_PER_WG) {
        float a = float(cut_loadA(dataA, pc, m, k));

        uint byteIdx = k * pc.strideBNpacked + (baseN >> 1);
        uint packed = packedB[byteIdx >> 2];
        float b0 = float(int((packed >>  0) & 0xFu) - 8);
        float b1 = float(int((packed >>  4) & 0xFu) - 8);
        float b2 = float(int((packed >>  8) & 0xFu) - 8);
        float b3 = float(int((packed >> 12) & 0xFu) - 8);
        float b4 = float(int((packed >> 16) & 0xFu) - 8);
        float b5 = float(int((packed >> 20) & 0xFu) - 8);
        float b6 = float(int((packed >> 24) & 0xFu) - 8);
        float b7 = float(int((packed >> 28) & 0xFu) - 8);

        uint scaleBase0 = (k >> 5) * pc.scaleStride + baseN;
        float s0_0, s1_0, s2_0, s3_0;
        cut_loadScale4(scalesB, scaleBase0, s0_0, s1_0, s2_0, s3_0);

        uint scaleBase1 = (k >> 5) * pc.scaleStride + baseN + 4;
        float s0_1, s1_1, s2_1, s3_1;
        cut_loadScale4(scalesB, scaleBase1, s0_1, s1_1, s2_1, s3_1);

        acc0 = mad(a, b0 * s0_0, acc0);
        acc1 = mad(a, b1 * s1_0, acc1);
        acc2 = mad(a, b2 * s2_0, acc2);
        acc3 = mad(a, b3 * s3_0, acc3);
        acc4 = mad(a, b4 * s0_1, acc4);
        acc5 = mad(a, b5 * s1_1, acc5);
        acc6 = mad(a, b6 * s2_1, acc6);
        acc7 = mad(a, b7 * s3_1, acc7);
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
        partial[wy][0] = acc0; partial[wy][1] = acc1;
        partial[wy][2] = acc2; partial[wy][3] = acc3;
        partial[wy][4] = acc4; partial[wy][5] = acc5;
        partial[wy][6] = acc6; partial[wy][7] = acc7;
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
