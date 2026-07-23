// MatMulQ4GemvKPar.cu
#include "MatMulQ4Common.cuh"

#define WG_SIZE 32
#define COLS_PER_WG 4

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

    for (uint k = tid; k < pc.K; k += WG_SIZE) {
        float a = float(cut_loadA(dataA, pc, m, k));

        uint byteIdx = k * pc.strideBNpacked + (baseN >> 1);
        uint packed = packedB[byteIdx >> 2];
        uint shift = (byteIdx & 3u) * 8u;
        uint twoBytes = (packed >> shift) & 0xFFFFu;

        float b0 = float(int((twoBytes >>  0) & 0xFu) - 8);
        float b1 = float(int((twoBytes >>  4) & 0xFu) - 8);
        float b2 = float(int((twoBytes >>  8) & 0xFu) - 8);
        float b3 = float(int((twoBytes >> 12) & 0xFu) - 8);

        uint scaleBase = (k >> 5) * pc.scaleStride + baseN;
        float s0, s1, s2, s3;
        cut_loadScale4(scalesB, scaleBase, s0, s1, s2, s3);
        float m0, m1, m2, m3;
        cut_loadMin4(scalesB, pc, scaleBase, m0, m1, m2, m3);

        acc0 = mad(a, b0 * s0 + m0, acc0);
        acc1 = mad(a, b1 * s1 + m1, acc1);
        acc2 = mad(a, b2 * s2 + m2, acc2);
        acc3 = mad(a, b3 * s3 + m3, acc3);
    }

    #pragma unroll
    for (uint offset = 16u; offset >= 1u; offset >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_xor_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_xor_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_xor_sync(0xffffffffu, acc3, offset);
    }

    if (tid == 0) {
        uint colCount = min(COLS_PER_WG, pc.N - baseN);
        cut_writeOutput(dataC, dataD, pc, m, baseN, acc0);
        if (colCount > 1) cut_writeOutput(dataC, dataD, pc, m, baseN + 1, acc1);
        if (colCount > 2) cut_writeOutput(dataC, dataD, pc, m, baseN + 2, acc2);
        if (colCount > 3) cut_writeOutput(dataC, dataD, pc, m, baseN + 3, acc3);
    }
}
