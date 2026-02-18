#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Naive matrix multiplication: no tiling, no shared memory.
// Each thread computes one output element by looping over the full K dimension.
// All reads go to global memory — shows the cost of redundant memory accesses.

struct PushConstants {
    uint M;  // rows of A
    uint K;  // cols of A / rows of B
    uint N;  // cols of B
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataA;

[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataB;

[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataC;

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint row = DTid.y;
    uint col = DTid.x;

    if (row >= pc.M || col >= pc.N) return;

    %SCALAR_DTYPE% sum = (%SCALAR_DTYPE%)(0);
    for (uint k = 0; k < pc.K; k++) {
        sum += dataA[row * pc.K + k] * dataB[k * pc.N + col];
    }

    dataC[row * pc.N + col] = sum;
}
