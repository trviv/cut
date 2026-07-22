#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// GEMV8M: small-M (2..16) matmul as a wave-per-row GEMV.
// C[M, N] = A[M, K] * B[K, N]
//
// Wave-per-row layout: the block is (32, ROWS_PER_WG) = 8 waves. Each wave
// computes the full-K dot product for its own row m = Gid.y*ROWS_PER_WG +
// GTid.y over the workgroup's 8 columns. All 8 waves read the SAME B elements
// (k = GTid.x, stride WG_SIZE), so B loads are served from L1/broadcast and
// the effective B traffic is ~one read per ceil(M/8) rows — unlike the naive
// shared-mem tile kernel, which runs at ~0.3 TFLOPS at prefill shapes like
// [15,576]x[576,960].
//
// Grid X: ceil(N/8) workgroups  (output columns)
// Grid Y: ceil(M/8) workgroups  (output rows)

#include "MatMulCommon.shaderh"

#define WG_SIZE 32
#define COLS_PER_WG 8
#define ROWS_PER_WG 8

// [wave][lane][column] partials — reduced entirely in shared memory so the
// kernel is correct for any wave size (wave32 NVIDIA, wave64 AMD).
groupshared %SCALAR_DTYPE_OUTPUT% partial[ROWS_PER_WG][WG_SIZE][COLS_PER_WG];

[numthreads(WG_SIZE, ROWS_PER_WG, 1)]
void main(uint3 GTid : SV_GroupThreadID,
          uint3 Gid  : SV_GroupID) {
    uint lane = GTid.x;   // 0..31, lane within the wave
    uint wy = GTid.y;     // 0..ROWS_PER_WG-1, which row this wave owns
    uint baseN = Gid.x * COLS_PER_WG;
    uint m = Gid.y * ROWS_PER_WG + wy;

    // Block-uniform (depends only on the group id), so the barrier below is
    // safe: either every thread returns or none does. Rows past M do NOT
    // return — they run the loop with a zero A so the barrier stays uniform.
    if (baseN >= pc.N) return;

    bool rowValid = (m < pc.M);

    %SCALAR_DTYPE_OUTPUT% acc0 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc1 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc2 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc3 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc4 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc5 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc6 = (%SCALAR_DTYPE_OUTPUT%)(0);
    %SCALAR_DTYPE_OUTPUT% acc7 = (%SCALAR_DTYPE_OUTPUT%)(0);

    uint vec4Col0 = baseN >> 2;
    uint vec4Col1 = (baseN + 4) >> 2;
    uint strideB4 = pc.strideB >> 2;

    for (uint k = lane; k < pc.K; k += WG_SIZE) {
        %SCALAR_DTYPE_OUTPUT% a = rowValid
            ? (%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k)
            : (%SCALAR_DTYPE_OUTPUT%)(0);
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

    // Reduce each wave's 32 lanes in shared memory (wave-size agnostic).
    partial[wy][lane][0] = acc0;
    partial[wy][lane][1] = acc1;
    partial[wy][lane][2] = acc2;
    partial[wy][lane][3] = acc3;
    partial[wy][lane][4] = acc4;
    partial[wy][lane][5] = acc5;
    partial[wy][lane][6] = acc6;
    partial[wy][lane][7] = acc7;
    GroupMemoryBarrierWithGroupSync();

    // Each wave folds only its own [wy] slice: the full-K dot product for its
    // row. No cross-wave fold is needed (unlike the split-K Gemv8).
    if (lane < COLS_PER_WG) {
        %SCALAR_DTYPE_OUTPUT% s = (%SCALAR_DTYPE_OUTPUT%)(0);
        [unroll] for (uint x = 0; x < WG_SIZE; ++x) s += partial[wy][x][lane];
        if (rowValid) {
            uint col = baseN + lane;
            if (col < pc.N) writeOutput(m, col, s);
        }
    }
}
