#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_SCALES%
%DTYPE_DEFINES_OUTPUT%

// Q8 GEMV with K-parallel subgroup reduction and packed uint32 dequantization.
// C[M, N] = A[M, K] * dequant(B[K, N])   (typically M=1)
//
// Improvements over MatMulQ8GemvKPar:
//   1. COLS_PER_WG = 8 (was 4) — halves WG launches
//   2. K-loop unrolled by 4 with explicit accumulator interleaving for ILP
//   3. Two packed uint32 loads for B per K step (8 columns = 2 groups of 4 int8)
//   4. Two scale loads per K step (2 groups of 4 scales)
//   5. Butterfly reduction across 32 lanes (no shared memory barriers)
//
// Grid X: ceil(N/8) workgroups  (output columns)
// Grid Y: M workgroups          (output rows, typically 1)

#include "MatMulQ8Common.shaderh"

#define WG_SIZE 32
#define COLS_PER_WG 8

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID,
          uint3 Gid  : SV_GroupID) {
    uint tid = GTid.x;
    uint baseN = Gid.x * COLS_PER_WG;
    uint m = Gid.y;

    if (baseN >= pc.N || m >= pc.M) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    // K-parallel: thread tid processes k = tid, tid+32, tid+64, ...
    // Unroll K-loop by 4 for ILP
    uint K4 = pc.K & ~(4u * WG_SIZE - 1u);
    uint k = tid;

    for (; k < K4; k += 4 * WG_SIZE) {
        // Iteration 0
        {
            float a = float(loadA_fast(m, k));

            uint byteIdx0 = k * pc.strideBN + baseN;
            uint packed0 = packedB[byteIdx0 >> 2];
            int sp0 = int(packed0);
            float b0_0 = float((sp0 << 24) >> 24);
            float b1_0 = float((sp0 << 16) >> 24);
            float b2_0 = float((sp0 <<  8) >> 24);
            float b3_0 = float(sp0 >> 24);

            uint byteIdx1 = k * pc.strideBN + baseN + 4;
            uint packed1 = packedB[byteIdx1 >> 2];
            int sp1 = int(packed1);
            float b0_1 = float((sp1 << 24) >> 24);
            float b1_1 = float((sp1 << 16) >> 24);
            float b2_1 = float((sp1 <<  8) >> 24);
            float b3_1 = float(sp1 >> 24);

            uint scaleBase0 = (k >> 5) * pc.scaleStride + baseN;
#ifdef DTYPE_SCALES_IS_INT8
            int packedScales0 = scalesB[scaleBase0 >> 4][(scaleBase0 >> 2) & 3];
            int ss0 = packedScales0;
            float s0_0 = float((ss0 << 24) >> 24);
            float s1_0 = float((ss0 << 16) >> 24);
            float s2_0 = float((ss0 <<  8) >> 24);
            float s3_0 = float(ss0 >> 24);
#else
            %VEC_DTYPE_SCALES% sv0 = scalesB[scaleBase0 >> 2];
            float s0_0 = float(sv0[0]);
            float s1_0 = float(sv0[1]);
            float s2_0 = float(sv0[2]);
            float s3_0 = float(sv0[3]);
#endif

            uint scaleBase1 = (k >> 5) * pc.scaleStride + baseN + 4;
#ifdef DTYPE_SCALES_IS_INT8
            int packedScales1 = scalesB[scaleBase1 >> 4][(scaleBase1 >> 2) & 3];
            int ss1 = packedScales1;
            float s0_1 = float((ss1 << 24) >> 24);
            float s1_1 = float((ss1 << 16) >> 24);
            float s2_1 = float((ss1 <<  8) >> 24);
            float s3_1 = float(ss1 >> 24);
#else
            %VEC_DTYPE_SCALES% sv1 = scalesB[scaleBase1 >> 2];
            float s0_1 = float(sv1[0]);
            float s1_1 = float(sv1[1]);
            float s2_1 = float(sv1[2]);
            float s3_1 = float(sv1[3]);
#endif

            acc0 = mad(a, b0_0 * s0_0, acc0);
            acc1 = mad(a, b1_0 * s1_0, acc1);
            acc2 = mad(a, b2_0 * s2_0, acc2);
            acc3 = mad(a, b3_0 * s3_0, acc3);
            acc4 = mad(a, b0_1 * s0_1, acc4);
            acc5 = mad(a, b1_1 * s1_1, acc5);
            acc6 = mad(a, b2_1 * s2_1, acc6);
            acc7 = mad(a, b3_1 * s3_1, acc7);
        }
        // Iteration 1
        {
            uint k1 = k + WG_SIZE;
            float a = float(loadA_fast(m, k1));

            uint byteIdx0 = k1 * pc.strideBN + baseN;
            uint packed0 = packedB[byteIdx0 >> 2];
            int sp0 = int(packed0);
            float b0_0 = float((sp0 << 24) >> 24);
            float b1_0 = float((sp0 << 16) >> 24);
            float b2_0 = float((sp0 <<  8) >> 24);
            float b3_0 = float(sp0 >> 24);

            uint byteIdx1 = k1 * pc.strideBN + baseN + 4;
            uint packed1 = packedB[byteIdx1 >> 2];
            int sp1 = int(packed1);
            float b0_1 = float((sp1 << 24) >> 24);
            float b1_1 = float((sp1 << 16) >> 24);
            float b2_1 = float((sp1 <<  8) >> 24);
            float b3_1 = float(sp1 >> 24);

            uint scaleBase0 = (k1 >> 5) * pc.scaleStride + baseN;
#ifdef DTYPE_SCALES_IS_INT8
            int packedScales0 = scalesB[scaleBase0 >> 4][(scaleBase0 >> 2) & 3];
            int ss0 = packedScales0;
            float s0_0 = float((ss0 << 24) >> 24);
            float s1_0 = float((ss0 << 16) >> 24);
            float s2_0 = float((ss0 <<  8) >> 24);
            float s3_0 = float(ss0 >> 24);
#else
            %VEC_DTYPE_SCALES% sv0 = scalesB[scaleBase0 >> 2];
            float s0_0 = float(sv0[0]);
            float s1_0 = float(sv0[1]);
            float s2_0 = float(sv0[2]);
            float s3_0 = float(sv0[3]);
#endif

            uint scaleBase1 = (k1 >> 5) * pc.scaleStride + baseN + 4;
#ifdef DTYPE_SCALES_IS_INT8
            int packedScales1 = scalesB[scaleBase1 >> 4][(scaleBase1 >> 2) & 3];
            int ss1 = packedScales1;
            float s0_1 = float((ss1 << 24) >> 24);
            float s1_1 = float((ss1 << 16) >> 24);
            float s2_1 = float((ss1 <<  8) >> 24);
            float s3_1 = float(ss1 >> 24);
#else
            %VEC_DTYPE_SCALES% sv1 = scalesB[scaleBase1 >> 2];
            float s0_1 = float(sv1[0]);
            float s1_1 = float(sv1[1]);
            float s2_1 = float(sv1[2]);
            float s3_1 = float(sv1[3]);
#endif

            acc0 = mad(a, b0_0 * s0_0, acc0);
            acc1 = mad(a, b1_0 * s1_0, acc1);
            acc2 = mad(a, b2_0 * s2_0, acc2);
            acc3 = mad(a, b3_0 * s3_0, acc3);
            acc4 = mad(a, b0_1 * s0_1, acc4);
            acc5 = mad(a, b1_1 * s1_1, acc5);
            acc6 = mad(a, b2_1 * s2_1, acc6);
            acc7 = mad(a, b3_1 * s3_1, acc7);
        }
        // Iteration 2
        {
            uint k2 = k + 2 * WG_SIZE;
            float a = float(loadA_fast(m, k2));

            uint byteIdx0 = k2 * pc.strideBN + baseN;
            uint packed0 = packedB[byteIdx0 >> 2];
            int sp0 = int(packed0);
            float b0_0 = float((sp0 << 24) >> 24);
            float b1_0 = float((sp0 << 16) >> 24);
            float b2_0 = float((sp0 <<  8) >> 24);
            float b3_0 = float(sp0 >> 24);

            uint byteIdx1 = k2 * pc.strideBN + baseN + 4;
            uint packed1 = packedB[byteIdx1 >> 2];
            int sp1 = int(packed1);
            float b0_1 = float((sp1 << 24) >> 24);
            float b1_1 = float((sp1 << 16) >> 24);
            float b2_1 = float((sp1 <<  8) >> 24);
            float b3_1 = float(sp1 >> 24);

            uint scaleBase0 = (k2 >> 5) * pc.scaleStride + baseN;
#ifdef DTYPE_SCALES_IS_INT8
            int packedScales0 = scalesB[scaleBase0 >> 4][(scaleBase0 >> 2) & 3];
            int ss0 = packedScales0;
            float s0_0 = float((ss0 << 24) >> 24);
            float s1_0 = float((ss0 << 16) >> 24);
            float s2_0 = float((ss0 <<  8) >> 24);
            float s3_0 = float(ss0 >> 24);
#else
            %VEC_DTYPE_SCALES% sv0 = scalesB[scaleBase0 >> 2];
            float s0_0 = float(sv0[0]);
            float s1_0 = float(sv0[1]);
            float s2_0 = float(sv0[2]);
            float s3_0 = float(sv0[3]);
#endif

            uint scaleBase1 = (k2 >> 5) * pc.scaleStride + baseN + 4;
#ifdef DTYPE_SCALES_IS_INT8
            int packedScales1 = scalesB[scaleBase1 >> 4][(scaleBase1 >> 2) & 3];
            int ss1 = packedScales1;
            float s0_1 = float((ss1 << 24) >> 24);
            float s1_1 = float((ss1 << 16) >> 24);
            float s2_1 = float((ss1 <<  8) >> 24);
            float s3_1 = float(ss1 >> 24);
#else
            %VEC_DTYPE_SCALES% sv1 = scalesB[scaleBase1 >> 2];
            float s0_1 = float(sv1[0]);
            float s1_1 = float(sv1[1]);
            float s2_1 = float(sv1[2]);
            float s3_1 = float(sv1[3]);
#endif

            acc0 = mad(a, b0_0 * s0_0, acc0);
            acc1 = mad(a, b1_0 * s1_0, acc1);
            acc2 = mad(a, b2_0 * s2_0, acc2);
            acc3 = mad(a, b3_0 * s3_0, acc3);
            acc4 = mad(a, b0_1 * s0_1, acc4);
            acc5 = mad(a, b1_1 * s1_1, acc5);
            acc6 = mad(a, b2_1 * s2_1, acc6);
            acc7 = mad(a, b3_1 * s3_1, acc7);
        }
        // Iteration 3
        {
            uint k3 = k + 3 * WG_SIZE;
            float a = float(loadA_fast(m, k3));

            uint byteIdx0 = k3 * pc.strideBN + baseN;
            uint packed0 = packedB[byteIdx0 >> 2];
            int sp0 = int(packed0);
            float b0_0 = float((sp0 << 24) >> 24);
            float b1_0 = float((sp0 << 16) >> 24);
            float b2_0 = float((sp0 <<  8) >> 24);
            float b3_0 = float(sp0 >> 24);

            uint byteIdx1 = k3 * pc.strideBN + baseN + 4;
            uint packed1 = packedB[byteIdx1 >> 2];
            int sp1 = int(packed1);
            float b0_1 = float((sp1 << 24) >> 24);
            float b1_1 = float((sp1 << 16) >> 24);
            float b2_1 = float((sp1 <<  8) >> 24);
            float b3_1 = float(sp1 >> 24);

            uint scaleBase0 = (k3 >> 5) * pc.scaleStride + baseN;
#ifdef DTYPE_SCALES_IS_INT8
            int packedScales0 = scalesB[scaleBase0 >> 4][(scaleBase0 >> 2) & 3];
            int ss0 = packedScales0;
            float s0_0 = float((ss0 << 24) >> 24);
            float s1_0 = float((ss0 << 16) >> 24);
            float s2_0 = float((ss0 <<  8) >> 24);
            float s3_0 = float(ss0 >> 24);
#else
            %VEC_DTYPE_SCALES% sv0 = scalesB[scaleBase0 >> 2];
            float s0_0 = float(sv0[0]);
            float s1_0 = float(sv0[1]);
            float s2_0 = float(sv0[2]);
            float s3_0 = float(sv0[3]);
#endif

            uint scaleBase1 = (k3 >> 5) * pc.scaleStride + baseN + 4;
#ifdef DTYPE_SCALES_IS_INT8
            int packedScales1 = scalesB[scaleBase1 >> 4][(scaleBase1 >> 2) & 3];
            int ss1 = packedScales1;
            float s0_1 = float((ss1 << 24) >> 24);
            float s1_1 = float((ss1 << 16) >> 24);
            float s2_1 = float((ss1 <<  8) >> 24);
            float s3_1 = float(ss1 >> 24);
#else
            %VEC_DTYPE_SCALES% sv1 = scalesB[scaleBase1 >> 2];
            float s0_1 = float(sv1[0]);
            float s1_1 = float(sv1[1]);
            float s2_1 = float(sv1[2]);
            float s3_1 = float(sv1[3]);
#endif

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

    // Remainder: single-stride K steps
    for (; k < pc.K; k += WG_SIZE) {
        float a = float(loadA_fast(m, k));

        uint byteIdx0 = k * pc.strideBN + baseN;
        uint packed0 = packedB[byteIdx0 >> 2];
        int sp0 = int(packed0);
        float b0_0 = float((sp0 << 24) >> 24);
        float b1_0 = float((sp0 << 16) >> 24);
        float b2_0 = float((sp0 <<  8) >> 24);
        float b3_0 = float(sp0 >> 24);

        uint byteIdx1 = k * pc.strideBN + baseN + 4;
        uint packed1 = packedB[byteIdx1 >> 2];
        int sp1 = int(packed1);
        float b0_1 = float((sp1 << 24) >> 24);
        float b1_1 = float((sp1 << 16) >> 24);
        float b2_1 = float((sp1 <<  8) >> 24);
        float b3_1 = float(sp1 >> 24);

        uint scaleBase0 = (k >> 5) * pc.scaleStride + baseN;
#ifdef DTYPE_SCALES_IS_INT8
        int packedScales0 = scalesB[scaleBase0 >> 4][(scaleBase0 >> 2) & 3];
        int ss0 = packedScales0;
        float s0_0 = float((ss0 << 24) >> 24);
        float s1_0 = float((ss0 << 16) >> 24);
        float s2_0 = float((ss0 <<  8) >> 24);
        float s3_0 = float(ss0 >> 24);
#else
        %VEC_DTYPE_SCALES% sv0 = scalesB[scaleBase0 >> 2];
        float s0_0 = float(sv0[0]);
        float s1_0 = float(sv0[1]);
        float s2_0 = float(sv0[2]);
        float s3_0 = float(sv0[3]);
#endif

        uint scaleBase1 = (k >> 5) * pc.scaleStride + baseN + 4;
#ifdef DTYPE_SCALES_IS_INT8
        int packedScales1 = scalesB[scaleBase1 >> 4][(scaleBase1 >> 2) & 3];
        int ss1 = packedScales1;
        float s0_1 = float((ss1 << 24) >> 24);
        float s1_1 = float((ss1 << 16) >> 24);
        float s2_1 = float((ss1 <<  8) >> 24);
        float s3_1 = float(ss1 >> 24);
#else
        %VEC_DTYPE_SCALES% sv1 = scalesB[scaleBase1 >> 2];
        float s0_1 = float(sv1[0]);
        float s1_1 = float(sv1[1]);
        float s2_1 = float(sv1[2]);
        float s3_1 = float(sv1[3]);
#endif

        acc0 = mad(a, b0_0 * s0_0, acc0);
        acc1 = mad(a, b1_0 * s1_0, acc1);
        acc2 = mad(a, b2_0 * s2_0, acc2);
        acc3 = mad(a, b3_0 * s3_0, acc3);
        acc4 = mad(a, b0_1 * s0_1, acc4);
        acc5 = mad(a, b1_1 * s1_1, acc5);
        acc6 = mad(a, b2_1 * s2_1, acc6);
        acc7 = mad(a, b3_1 * s3_1, acc7);
    }

    // Butterfly reduction across 32 lanes (no barriers needed)
    [unroll] for (uint offset = 16; offset >= 1; offset >>= 1) {
        acc0 += WaveReadLaneAt(acc0, tid ^ offset);
        acc1 += WaveReadLaneAt(acc1, tid ^ offset);
        acc2 += WaveReadLaneAt(acc2, tid ^ offset);
        acc3 += WaveReadLaneAt(acc3, tid ^ offset);
        acc4 += WaveReadLaneAt(acc4, tid ^ offset);
        acc5 += WaveReadLaneAt(acc5, tid ^ offset);
        acc6 += WaveReadLaneAt(acc6, tid ^ offset);
        acc7 += WaveReadLaneAt(acc7, tid ^ offset);
    }

    // Lane 0 writes the final results
    if (tid == 0) {
        uint colCount = min(COLS_PER_WG, pc.N - baseN);
        writeOutput(m, baseN, acc0);
        if (colCount > 1) writeOutput(m, baseN + 1, acc1);
        if (colCount > 2) writeOutput(m, baseN + 2, acc2);
        if (colCount > 3) writeOutput(m, baseN + 3, acc3);
        if (colCount > 4) writeOutput(m, baseN + 4, acc4);
        if (colCount > 5) writeOutput(m, baseN + 5, acc5);
        if (colCount > 6) writeOutput(m, baseN + 6, acc6);
        if (colCount > 7) writeOutput(m, baseN + 7, acc7);
    }
}
