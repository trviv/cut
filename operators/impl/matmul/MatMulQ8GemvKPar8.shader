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
#define WARPS_PER_WG 8

// [wave][lane][column] partials — reduced entirely in shared memory so
// the kernel is correct for any wave size (wave32 NVIDIA, wave64 AMD).
groupshared float partial[WARPS_PER_WG][WG_SIZE][COLS_PER_WG];

[numthreads(WG_SIZE, WARPS_PER_WG, 1)]
void main(uint3 GTid : SV_GroupThreadID,
          uint3 Gid  : SV_GroupID) {
    uint lane = GTid.x;   // 0..31 within the wave
    uint wy = GTid.y;     // 0..7, which K stripe this wave owns
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

    // Split-K: wave wy walks the stripe k = wy*32 + lane, stride 32*8.
    // The 8 independent accumulators supply the ILP the manual unroll gave.
    for (uint k = wy * WG_SIZE + lane; k < pc.K; k += WG_SIZE * WARPS_PER_WG) {
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

    partial[wy][lane][0] = acc0;
    partial[wy][lane][1] = acc1;
    partial[wy][lane][2] = acc2;
    partial[wy][lane][3] = acc3;
    partial[wy][lane][4] = acc4;
    partial[wy][lane][5] = acc5;
    partial[wy][lane][6] = acc6;
    partial[wy][lane][7] = acc7;
    GroupMemoryBarrierWithGroupSync();

    // Stage 1: for each K stripe, 8 threads fold that stripe's 32 lanes.
    if (lane < COLS_PER_WG) {
        float s = (float)(0);
        [unroll] for (uint x = 0; x < WG_SIZE; ++x) s += partial[wy][x][lane];
        partial[wy][0][lane] = s;
    }
    GroupMemoryBarrierWithGroupSync();

    // Stage 2: wave 0 folds the per-stripe sums into the final column sums.
    if (wy == 0 && lane < COLS_PER_WG) {
        float sum = (float)(0);
        [unroll] for (uint w = 0; w < WARPS_PER_WG; ++w) sum += partial[w][0][lane];
        uint col = baseN + lane;
        if (col < pc.N) writeOutput(m, col, sum);
    }
}
