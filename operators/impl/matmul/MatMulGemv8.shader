#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// GEMV8: matrix-vector multiply optimized for M=1 (autoregressive decoding).
// C[M, N] = A[M, K] * B[K, N]   (typically M=1)
//
// Improvements over MatMulGemv:
//   1. Processes 8 output columns per workgroup (vs 4) — halves WG launches
//   2. K-loop unrolled by 4 with explicit accumulator interleaving for ILP
//   3. Two vec4 B loads per K-step (8 columns)
//   4. Butterfly reduction across 32 lanes (no shared memory barriers)
//
// Grid X: ceil(N/8) workgroups  (output columns)
// Grid Y: M workgroups          (output rows, typically 1)

#include "MatMulCommon.shaderh"

#define WG_SIZE 32
#define COLS_PER_WG 8

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID,
          uint3 Gid  : SV_GroupID) {
    uint tid = GTid.x;
    uint baseN = Gid.x * COLS_PER_WG;
    uint m = Gid.y;  // row of A / row of output C

    if (baseN >= pc.N || m >= pc.M) return;

    // Accumulators for 8 output columns (two groups of 4)
    %SCALAR_DTYPE_OUTPUT% acc0 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc1 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc2 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc3 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc4 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc5 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc6 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc7 = (%SCALAR_DTYPE_OUTPUT%)(0);

    // Vec4 index for the two groups of 4 columns
    uint vec4Col0 = baseN >> 2;
    uint vec4Col1 = (baseN + 4) >> 2;
    uint strideB4 = pc.strideB >> 2;

    // K-parallel: thread tid processes k = tid, tid+32, tid+64, ...
    // Unroll K-loop by 4 for ILP
    uint K4 = pc.K & ~(4u * WG_SIZE - 1u);
    uint k = tid;

    for (; k < K4; k += 4 * WG_SIZE) {
        // Iteration 0
        {
            %SCALAR_DTYPE_OUTPUT% a = (%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k);
            %VEC_DTYPE_INPUT2% bVec0 = dataB[k * strideB4 + vec4Col0];
            %VEC_DTYPE_INPUT2% bVec1 = dataB[k * strideB4 + vec4Col1];
            acc0 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[0], acc0);
            acc1 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[1], acc1);
            acc2 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[2], acc2);
            acc3 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[3], acc3);
            acc4 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[0], acc4);
            acc5 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[1], acc5);
            acc6 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[2], acc6);
            acc7 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[3], acc7);
        }
        // Iteration 1
        {
            uint k1 = k + WG_SIZE;
            %SCALAR_DTYPE_OUTPUT% a = (%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k1);
            %VEC_DTYPE_INPUT2% bVec0 = dataB[k1 * strideB4 + vec4Col0];
            %VEC_DTYPE_INPUT2% bVec1 = dataB[k1 * strideB4 + vec4Col1];
            acc0 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[0], acc0);
            acc1 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[1], acc1);
            acc2 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[2], acc2);
            acc3 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[3], acc3);
            acc4 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[0], acc4);
            acc5 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[1], acc5);
            acc6 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[2], acc6);
            acc7 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[3], acc7);
        }
        // Iteration 2
        {
            uint k2 = k + 2 * WG_SIZE;
            %SCALAR_DTYPE_OUTPUT% a = (%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k2);
            %VEC_DTYPE_INPUT2% bVec0 = dataB[k2 * strideB4 + vec4Col0];
            %VEC_DTYPE_INPUT2% bVec1 = dataB[k2 * strideB4 + vec4Col1];
            acc0 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[0], acc0);
            acc1 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[1], acc1);
            acc2 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[2], acc2);
            acc3 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[3], acc3);
            acc4 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[0], acc4);
            acc5 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[1], acc5);
            acc6 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[2], acc6);
            acc7 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[3], acc7);
        }
        // Iteration 3
        {
            uint k3 = k + 3 * WG_SIZE;
            %SCALAR_DTYPE_OUTPUT% a = (%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k3);
            %VEC_DTYPE_INPUT2% bVec0 = dataB[k3 * strideB4 + vec4Col0];
            %VEC_DTYPE_INPUT2% bVec1 = dataB[k3 * strideB4 + vec4Col1];
            acc0 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[0], acc0);
            acc1 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[1], acc1);
            acc2 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[2], acc2);
            acc3 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[3], acc3);
            acc4 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[0], acc4);
            acc5 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[1], acc5);
            acc6 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[2], acc6);
            acc7 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[3], acc7);
        }
    }

    // Remainder: single-stride K steps
    for (; k < pc.K; k += WG_SIZE) {
        %SCALAR_DTYPE_OUTPUT% a = (%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k);
        %VEC_DTYPE_INPUT2% bVec0 = dataB[k * strideB4 + vec4Col0];
        %VEC_DTYPE_INPUT2% bVec1 = dataB[k * strideB4 + vec4Col1];
        acc0 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[0], acc0);
        acc1 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[1], acc1);
        acc2 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[2], acc2);
        acc3 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec0[3], acc3);
        acc4 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[0], acc4);
        acc5 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[1], acc5);
        acc6 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[2], acc6);
        acc7 = mad(a, (%SCALAR_DTYPE_OUTPUT%)bVec1[3], acc7);
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
