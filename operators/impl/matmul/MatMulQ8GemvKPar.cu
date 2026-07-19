// MatMulQ8GemvKPar.cu
#include "MatMulQ8Common.cuh"

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
        float a = float(cut_loadA_fast(dataA, pc, m, k));
        uint byteIdx = k * pc.strideBN + baseN;
        uint packed = packedB[byteIdx >> 2];
        float b0, b1, b2, b3;
        cut_unpackB4(packed, b0, b1, b2, b3);
        uint scaleBase = (k >> 5) * pc.scaleStride + baseN;
        float s0, s1, s2, s3;
        cut_loadScale4(scalesB, scaleBase, s0, s1, s2, s3);
        acc0 = mad(a, b0 * s0, acc0);
        acc1 = mad(a, b1 * s1, acc1);
        acc2 = mad(a, b2 * s2, acc2);
        acc3 = mad(a, b3 * s3, acc3);
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
