#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_SCALES%
%DTYPE_DEFINES_OUTPUT%

// Q4 GEMV with K-parallel subgroup reduction and nibble unpacking.
// C[M, N] = A[M, K] * dequant(B[K, N])   (typically M=1)
//
// Fixes vs MatMulQ4Gemv:
//   1. K-parallel: 32 threads split K work, reduced via WaveReadLaneAt
//      (old: each thread loops ALL K = no parallelism)
//   2. Nibble unpacking: extract 4 nibbles per load (4x fewer B loads)
//      (old: individual nibble extraction per element)
//   3. Vec4 scale loading: one load for 4 column scales
//      (old: one scalar scale load per element)
//
// Grid X: ceil(N/4) workgroups  (output columns, 4 per WG)
// Grid Y: M workgroups          (output rows, typically 1)

#include "MatMulQ4Common.shaderh"

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
        float a = float(loadA(m, k));

        // Load 4 nibbles for baseN..baseN+3 from packed B [K, N/2]
        // baseN is 4-aligned (Gid.x * 4), so baseN/2 is 2-aligned.
        uint byteIdx = k * pc.strideBNpacked + (baseN >> 1);
        uint packed = packedB[byteIdx >> 2];
        uint shift = (byteIdx & 3u) * 8u;
        uint twoBytes = (packed >> shift) & 0xFFFFu;

        // Extract 4 nibbles with Q4_0 dequantization (nibble - 8)
        float b0 = float(int((twoBytes >>  0) & 0xFu) - 8);
        float b1 = float(int((twoBytes >>  4) & 0xFu) - 8);
        float b2 = float(int((twoBytes >>  8) & 0xFu) - 8);
        float b3 = float(int((twoBytes >> 12) & 0xFu) - 8);

        // Load 4 scales for this K-block (one per column)
        // scaleBase is 4-aligned → one vec4 load for 4 scales
        uint scaleBase = (k >> 5) * pc.scaleStride + baseN;
        %VEC_DTYPE_SCALES% sv = scalesB[scaleBase >> 2];
        float s0 = float(sv[0]);
        float s1 = float(sv[1]);
        float s2 = float(sv[2]);
        float s3 = float(sv[3]);

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
