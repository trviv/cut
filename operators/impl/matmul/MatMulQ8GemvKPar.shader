#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_SCALES%
%DTYPE_DEFINES_OUTPUT%

// Q8 GEMV with K-parallel subgroup reduction and packed uint32 dequantization.
// C[M, N] = A[M, K] * dequant(B[K, N])   (typically M=1)
//
// Fixes vs MatMulQ8Gemv:
//   1. K-parallel: 32 threads split K work, reduced via WaveReadLaneAt
//      (old: each thread loops ALL K = no parallelism)
//   2. Packed uint32 dequant: unpack 4 int8 values per load (4x fewer B loads)
//      (old: individual byte extraction per element)
//   3. Vec4 scale loading: one load for 4 column scales
//      (old: one scalar scale load per element)
//
// Grid X: ceil(N/4) workgroups  (output columns, 4 per WG)
// Grid Y: M workgroups          (output rows, typically 1)

#include "MatMulQ8Common.shaderh"

#define WG_SIZE 32
#define COLS_PER_WG 4

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

    // K-parallel: thread tid processes k = tid, tid+32, tid+64, ...
    for (uint k = tid; k < pc.K; k += WG_SIZE) {
        float a = float(loadA_fast(m, k));

        // Packed uint32 load: 4 adjacent int8 values at B[k, baseN..baseN+3]
        // baseN is always 4-aligned (Gid.x * 4), strideBN is 4-aligned,
        // so byteIdx is 4-aligned and maps to one clean uint32.
        uint byteIdx = k * pc.strideBN + baseN;
        uint packed = packedB[byteIdx >> 2];

        // Unpack 4 int8 values with sign extension
        int sp = int(packed);
        float b0 = float((sp << 24) >> 24);
        float b1 = float((sp << 16) >> 24);
        float b2 = float((sp <<  8) >> 24);
        float b3 = float(sp >> 24);

        // Load 4 scales for this K-block (one per column)
        // scaleBase is 4-aligned → one vec4 load for 4 scales
        uint scaleBase = (k >> 5) * pc.scaleStride + baseN;
#ifdef DTYPE_SCALES_IS_INT8
        int packedScales = scalesB[scaleBase >> 4][(scaleBase >> 2) & 3];
        int ss = packedScales;
        float s0 = float((ss << 24) >> 24);
        float s1 = float((ss << 16) >> 24);
        float s2 = float((ss <<  8) >> 24);
        float s3 = float(ss >> 24);
#else
        %VEC_DTYPE_SCALES% sv = scalesB[scaleBase >> 2];
        float s0 = float(sv[0]);
        float s1 = float(sv[1]);
        float s2 = float(sv[2]);
        float s3 = float(sv[3]);
#endif

        // Dequantize and accumulate
        acc0 = mad(a, b0 * s0, acc0);
        acc1 = mad(a, b1 * s1, acc1);
        acc2 = mad(a, b2 * s2, acc2);
        acc3 = mad(a, b3 * s3, acc3);
    }

    // Butterfly reduction across 32 lanes (no barriers needed)
    [unroll] for (uint offset = 16; offset >= 1; offset >>= 1) {
        acc0 += WaveReadLaneAt(acc0, tid ^ offset);
        acc1 += WaveReadLaneAt(acc1, tid ^ offset);
        acc2 += WaveReadLaneAt(acc2, tid ^ offset);
        acc3 += WaveReadLaneAt(acc3, tid ^ offset);
    }

    // Lane 0 writes the final results
    if (tid == 0) {
        uint colCount = min(COLS_PER_WG, pc.N - baseN);
        writeOutput(m, baseN, acc0);
        if (colCount > 1) writeOutput(m, baseN + 1, acc1);
        if (colCount > 2) writeOutput(m, baseN + 2, acc2);
        if (colCount > 3) writeOutput(m, baseN + 3, acc3);
    }
}
