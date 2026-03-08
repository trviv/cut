#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Naive matrix multiplication: no tiling, no shared memory.
// Each thread computes one output element by looping over the full K dimension.
// All reads go to global memory — shows the cost of redundant memory accesses.

#include "MatMulCommon.shaderh"

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint row = DTid.y;
    uint col = DTid.x;

    if (row >= pc.M || col >= pc.N) return;

    %SCALAR_DTYPE_OUTPUT% sum = (%SCALAR_DTYPE_OUTPUT%)(0);
    for (uint k = 0; k < pc.K; k++) {
        uint idxA = row * pc.strideA + k;
        uint idxB = k * pc.strideB + col;
        sum += (%SCALAR_DTYPE_OUTPUT%)(dataA[idxA >> 2][idxA & 3]) * (%SCALAR_DTYPE_OUTPUT%)(dataB[idxB >> 2][idxB & 3]);
    }

    writeOutput(row, col, sum);
}
