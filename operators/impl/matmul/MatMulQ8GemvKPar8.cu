// MatMulQ8GemvKPar8.cu
#include "MatMulQ8Common.cuh"

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
        {
            uint kX = k;
            float a = float(cut_loadA_fast(dataA, pc, m, kX));
            uint byteIdx0 = kX * pc.strideBN + baseN;
            uint packed0 = packedB[byteIdx0 >> 2];
            float b0_0, b1_0, b2_0, b3_0;
            cut_unpackB4(packed0, b0_0, b1_0, b2_0, b3_0);
            uint byteIdx1 = kX * pc.strideBN + baseN + 4;
            uint packed1 = packedB[byteIdx1 >> 2];
            float b0_1, b1_1, b2_1, b3_1;
            cut_unpackB4(packed1, b0_1, b1_1, b2_1, b3_1);
            uint scaleBase0 = (kX >> 5) * pc.scaleStride + baseN;
            float s0_0, s1_0, s2_0, s3_0;
            cut_loadScale4(scalesB, scaleBase0, s0_0, s1_0, s2_0, s3_0);
            uint scaleBase1 = (kX >> 5) * pc.scaleStride + baseN + 4;
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
        {
            uint kX = k + WG_SIZE;
            float a = float(cut_loadA_fast(dataA, pc, m, kX));
            uint byteIdx0 = kX * pc.strideBN + baseN;
            uint packed0 = packedB[byteIdx0 >> 2];
            float b0_0, b1_0, b2_0, b3_0;
            cut_unpackB4(packed0, b0_0, b1_0, b2_0, b3_0);
            uint byteIdx1 = kX * pc.strideBN + baseN + 4;
            uint packed1 = packedB[byteIdx1 >> 2];
            float b0_1, b1_1, b2_1, b3_1;
            cut_unpackB4(packed1, b0_1, b1_1, b2_1, b3_1);
            uint scaleBase0 = (kX >> 5) * pc.scaleStride + baseN;
            float s0_0, s1_0, s2_0, s3_0;
            cut_loadScale4(scalesB, scaleBase0, s0_0, s1_0, s2_0, s3_0);
            uint scaleBase1 = (kX >> 5) * pc.scaleStride + baseN + 4;
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
        {
            uint kX = k + 2 * WG_SIZE;
            float a = float(cut_loadA_fast(dataA, pc, m, kX));
            uint byteIdx0 = kX * pc.strideBN + baseN;
            uint packed0 = packedB[byteIdx0 >> 2];
            float b0_0, b1_0, b2_0, b3_0;
            cut_unpackB4(packed0, b0_0, b1_0, b2_0, b3_0);
            uint byteIdx1 = kX * pc.strideBN + baseN + 4;
            uint packed1 = packedB[byteIdx1 >> 2];
            float b0_1, b1_1, b2_1, b3_1;
            cut_unpackB4(packed1, b0_1, b1_1, b2_1, b3_1);
            uint scaleBase0 = (kX >> 5) * pc.scaleStride + baseN;
            float s0_0, s1_0, s2_0, s3_0;
            cut_loadScale4(scalesB, scaleBase0, s0_0, s1_0, s2_0, s3_0);
            uint scaleBase1 = (kX >> 5) * pc.scaleStride + baseN + 4;
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
        {
            uint kX = k + 3 * WG_SIZE;
            float a = float(cut_loadA_fast(dataA, pc, m, kX));
            uint byteIdx0 = kX * pc.strideBN + baseN;
            uint packed0 = packedB[byteIdx0 >> 2];
            float b0_0, b1_0, b2_0, b3_0;
            cut_unpackB4(packed0, b0_0, b1_0, b2_0, b3_0);
            uint byteIdx1 = kX * pc.strideBN + baseN + 4;
            uint packed1 = packedB[byteIdx1 >> 2];
            float b0_1, b1_1, b2_1, b3_1;
            cut_unpackB4(packed1, b0_1, b1_1, b2_1, b3_1);
            uint scaleBase0 = (kX >> 5) * pc.scaleStride + baseN;
            float s0_0, s1_0, s2_0, s3_0;
            cut_loadScale4(scalesB, scaleBase0, s0_0, s1_0, s2_0, s3_0);
            uint scaleBase1 = (kX >> 5) * pc.scaleStride + baseN + 4;
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
    }

    for (; k < pc.K; k += WG_SIZE) {
        uint kX = k;
        float a = float(cut_loadA_fast(dataA, pc, m, kX));
        uint byteIdx0 = kX * pc.strideBN + baseN;
        uint packed0 = packedB[byteIdx0 >> 2];
        float b0_0, b1_0, b2_0, b3_0;
        cut_unpackB4(packed0, b0_0, b1_0, b2_0, b3_0);
        uint byteIdx1 = kX * pc.strideBN + baseN + 4;
        uint packed1 = packedB[byteIdx1 >> 2];
        float b0_1, b1_1, b2_1, b3_1;
        cut_unpackB4(packed1, b0_1, b1_1, b2_1, b3_1);
        uint scaleBase0 = (kX >> 5) * pc.scaleStride + baseN;
        float s0_0, s1_0, s2_0, s3_0;
        cut_loadScale4(scalesB, scaleBase0, s0_0, s1_0, s2_0, s3_0);
        uint scaleBase1 = (kX >> 5) * pc.scaleStride + baseN + 4;
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
