#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// GEMV8: matrix-vector multiply optimized for M=1 (autoregressive decoding).
// C[M, N] = A[M, K] * B[K, N]   (typically M=1)
//
// Split-K layout: the block is (32, WARPS_PER_WG) = 8 waves. All waves compute
// the SAME 8 output columns but each owns a different K stripe, so the grid is
// unchanged (ceil(N/8) workgroups) while 8x more waves are resident. The
// previous one-wave-per-group version launched only ceil(N/8) waves total — for
// a decode matmul with N=576 that is 72 waves on an 82-SM GPU (0.9 waves/SM
// against a 48-64 wave capacity), so memory latency was never hidden and the
// kernel ran at ~12% of peak bandwidth. Each wave keeps 8 independent
// accumulators (ILP), reduces across its 32 lanes with a butterfly shuffle,
// publishes one partial per column to shared memory, and wave 0 folds the 8
// partials into the final sums.
//
// Grid X: ceil(N/8) workgroups  (output columns)
// Grid Y: M workgroups          (output rows, typically 1)

#include "MatMulCommon.shaderh"

#define WG_SIZE 32
#define COLS_PER_WG 8
#define WARPS_PER_WG 8

// [wave][lane][column] partials — reduced entirely in shared memory so
// the kernel is correct for any wave size (wave32 NVIDIA, wave64 AMD).
groupshared %SCALAR_DTYPE_OUTPUT% partial[WARPS_PER_WG][WG_SIZE][COLS_PER_WG];

[numthreads(WG_SIZE, WARPS_PER_WG, 1)]
void main(uint3 GTid : SV_GroupThreadID,
          uint3 Gid  : SV_GroupID) {
    uint lane = GTid.x;   // 0..31, lane within the wave
    uint wy = GTid.y;     // 0..WARPS_PER_WG-1, which K stripe this wave owns
    uint baseN = Gid.x * COLS_PER_WG;
    uint m = Gid.y;       // row of A / row of output C

    // Block-uniform (depends only on the group id), so the barrier below is
    // safe: either every thread returns or none does.
    if (baseN >= pc.N || m >= pc.M) return;

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

    for (uint k = wy * WG_SIZE + lane; k < pc.K; k += WG_SIZE * WARPS_PER_WG) {
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

    // Reduce the 32 lanes within each wave.

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
        %SCALAR_DTYPE_OUTPUT% s = (%SCALAR_DTYPE_OUTPUT%)(0);
        [unroll] for (uint x = 0; x < WG_SIZE; ++x) s += partial[wy][x][lane];
        partial[wy][0][lane] = s;
    }
    GroupMemoryBarrierWithGroupSync();

    // Stage 2: wave 0 folds the per-stripe sums into the final column sums.
    if (wy == 0 && lane < COLS_PER_WG) {
        %SCALAR_DTYPE_OUTPUT% sum = (%SCALAR_DTYPE_OUTPUT%)(0);
        [unroll] for (uint w = 0; w < WARPS_PER_WG; ++w) sum += partial[w][0][lane];
        uint col = baseN + lane;
        if (col < pc.N) writeOutput(m, col, sum);
    }
}
