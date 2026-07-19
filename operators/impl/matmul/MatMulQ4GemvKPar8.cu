// MatMulQ4GemvKPar8.cu
#include "MatMulQ4Common.cuh"

#define WG_SIZE 32
#define COLS_PER_WG 8

extern "C" __global__ void cut_main(
    const CUT_VEC_DTYPE_INPUT1* __restrict__ dataA,
    const uint* __restrict__ packedB,
    const CUT_VEC_DTYPE_SCALES* __restrict__ scalesB,
    const CUT_VEC_DTYPE_OUTPUT* __restrict__ dataD,
    CUT_SCALAR_DTYPE_OUTPUT* __restrict__ dataC,
    PushConstants pc) {

    uint tid = threadIdx.x;
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

    uint K4 = pc.K & ~(4u * WG_SIZE - 1u);
    uint k = tid;

    for (; k < K4; k += 4 * WG_SIZE) {
        // Iteration 0
        {
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
        // Iteration 1
        {
            uint k1 = k + WG_SIZE;
            float a = float(cut_loadA(dataA, pc, m, k1));

            uint byteIdx = k1 * pc.strideBNpacked + (baseN >> 1);
            uint packed = packedB[byteIdx >> 2];
            float b0 = float(int((packed >>  0) & 0xFu) - 8);
            float b1 = float(int((packed >>  4) & 0xFu) - 8);
            float b2 = float(int((packed >>  8) & 0xFu) - 8);
            float b3 = float(int((packed >> 12) & 0xFu) - 8);
            float b4 = float(int((packed >> 16) & 0xFu) - 8);
            float b5 = float(int((packed >> 20) & 0xFu) - 8);
            float b6 = float(int((packed >> 24) & 0xFu) - 8);
            float b7 = float(int((packed >> 28) & 0xFu) - 8);

            uint scaleBase0 = (k1 >> 5) * pc.scaleStride + baseN;
            float s0_0, s1_0, s2_0, s3_0;
            cut_loadScale4(scalesB, scaleBase0, s0_0, s1_0, s2_0, s3_0);

            uint scaleBase1 = (k1 >> 5) * pc.scaleStride + baseN + 4;
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
        // Iteration 2
        {
            uint k2 = k + 2 * WG_SIZE;
            float a = float(cut_loadA(dataA, pc, m, k2));

            uint byteIdx = k2 * pc.strideBNpacked + (baseN >> 1);
            uint packed = packedB[byteIdx >> 2];
            float b0 = float(int((packed >>  0) & 0xFu) - 8);
            float b1 = float(int((packed >>  4) & 0xFu) - 8);
            float b2 = float(int((packed >>  8) & 0xFu) - 8);
            float b3 = float(int((packed >> 12) & 0xFu) - 8);
            float b4 = float(int((packed >> 16) & 0xFu) - 8);
            float b5 = float(int((packed >> 20) & 0xFu) - 8);
            float b6 = float(int((packed >> 24) & 0xFu) - 8);
            float b7 = float(int((packed >> 28) & 0xFu) - 8);

            uint scaleBase0 = (k2 >> 5) * pc.scaleStride + baseN;
            float s0_0, s1_0, s2_0, s3_0;
            cut_loadScale4(scalesB, scaleBase0, s0_0, s1_0, s2_0, s3_0);

            uint scaleBase1 = (k2 >> 5) * pc.scaleStride + baseN + 4;
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
        // Iteration 3
        {
            uint k3 = k + 3 * WG_SIZE;
            float a = float(cut_loadA(dataA, pc, m, k3));

            uint byteIdx = k3 * pc.strideBNpacked + (baseN >> 1);
            uint packed = packedB[byteIdx >> 2];
            float b0 = float(int((packed >>  0) & 0xFu) - 8);
            float b1 = float(int((packed >>  4) & 0xFu) - 8);
            float b2 = float(int((packed >>  8) & 0xFu) - 8);
            float b3 = float(int((packed >> 12) & 0xFu) - 8);
            float b4 = float(int((packed >> 16) & 0xFu) - 8);
            float b5 = float(int((packed >> 20) & 0xFu) - 8);
            float b6 = float(int((packed >> 24) & 0xFu) - 8);
            float b7 = float(int((packed >> 28) & 0xFu) - 8);

            uint scaleBase0 = (k3 >> 5) * pc.scaleStride + baseN;
            float s0_0, s1_0, s2_0, s3_0;
            cut_loadScale4(scalesB, scaleBase0, s0_0, s1_0, s2_0, s3_0);

            uint scaleBase1 = (k3 >> 5) * pc.scaleStride + baseN + 4;
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
    }

    for (; k < pc.K; k += WG_SIZE) {
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

    if (tid == 0) {
        uint colCount = min(COLS_PER_WG, pc.N - baseN);
        cut_writeOutput(dataC, dataD, pc, m, baseN, acc0);
        if (colCount > 1) cut_writeOutput(dataC, dataD, pc, m, baseN + 1, acc1);
        if (colCount > 2) cut_writeOutput(dataC, dataD, pc, m, baseN + 2, acc2);
        if (colCount > 3) cut_writeOutput(dataC, dataD, pc, m, baseN + 3, acc3);
        if (colCount > 4) cut_writeOutput(dataC, dataD, pc, m, baseN + 4, acc4);
        if (colCount > 5) cut_writeOutput(dataC, dataD, pc, m, baseN + 5, acc5);
        if (colCount > 6) cut_writeOutput(dataC, dataD, pc, m, baseN + 6, acc6);
        if (colCount > 7) cut_writeOutput(dataC, dataD, pc, m, baseN + 7, acc7);
    }
}
