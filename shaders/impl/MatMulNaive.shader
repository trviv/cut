#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Naive matrix multiplication: no tiling, no shared memory.
// Each thread computes one output element by looping over the full K dimension.
// All reads go to global memory — shows the cost of redundant memory accesses.

struct PushConstants {
    uint M;  // rows of A
    uint K;  // cols of A / rows of B
    uint N;  // cols of B
    uint strideA; // padded K (multiple of 4)
    uint strideB; // padded N (multiple of 4)
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE%> dataA;
[[vk::binding(1, 0)]] StructuredBuffer<%VEC_DTYPE%> dataB;
[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataC;

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint row = DTid.y;
    uint col = DTid.x;

    if (row >= pc.M || col >= pc.N) return;

    %SCALAR_DTYPE% sum = (%SCALAR_DTYPE%)(0);
    for (uint k = 0; k < pc.K; k++) {
        uint idxA = row * pc.strideA + k;
        uint idxB = k * pc.strideB + col;
        sum += dataA[idxA >> 2][idxA & 3] * dataB[idxB >> 2][idxB & 3];
    }

    dataC[row * pc.strideB + col] = sum;
}
