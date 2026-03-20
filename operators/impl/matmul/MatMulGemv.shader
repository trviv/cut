#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// GEMV: matrix-vector multiply optimized for M=1 (autoregressive decoding).
// C[M, N] = A[M, K] * B[K, N]   (typically M=1)
//
// K-parallel with subgroup reduction (no shared memory barriers).
//
// Each workgroup = 1 subgroup (32 threads) computing 4 output columns for
// one row of A.  K is split across 32 threads, reduced via WaveReadLaneAt.
// B is loaded as vec4 (4 adjacent columns) per thread per K-step.
//
// Grid X: ceil(N/4) workgroups  (output columns)
// Grid Y: M workgroups          (output rows, typically 1)

#include "MatMulCommon.shaderh"

#define WG_SIZE 32
#define COLS_PER_WG 4

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID,
          uint3 Gid  : SV_GroupID) {
    uint tid = GTid.x;
    uint baseN = Gid.x * COLS_PER_WG;
    uint m = Gid.y;  // row of A / row of output C

    if (baseN >= pc.N || m >= pc.M) return;

    // Accumulators for 4 output columns
    %SCALAR_DTYPE_OUTPUT% acc0 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc1 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc2 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc3 = (%SCALAR_DTYPE_OUTPUT%)(0);

    // K-parallel: thread tid processes k = tid, tid+32, tid+64, ...
    // Vec4 B load reads 4 adjacent columns per K-step.
    uint vec4Col = baseN >> 2;
    uint strideB4 = pc.strideB >> 2;

    for (uint k = tid; k < pc.K; k += WG_SIZE) {
        %SCALAR_DTYPE_OUTPUT% a = (%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k);
        %VEC_DTYPE_INPUT2% bVec = dataB[k * strideB4 + vec4Col];
        acc0 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec[0], acc0);
        acc1 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec[1], acc1);
        acc2 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec[2], acc2);
        acc3 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec[3], acc3);
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
