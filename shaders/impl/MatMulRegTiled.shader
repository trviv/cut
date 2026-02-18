#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Register-tiled matrix multiplication: each thread computes a TM x TN block
// of the output using only registers (local variables). No shared memory.
// Demonstrates register-level data reuse without cooperative loading.

#define TM 4
#define TN 4

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
    uint baseRow = DTid.y * TM;
    uint baseCol = DTid.x * TN;

    // Accumulator registers for TM x TN output block
    %SCALAR_DTYPE% acc[TM][TN];
    [unroll] for (uint m = 0; m < TM; m++)
        [unroll] for (uint n = 0; n < TN; n++)
            acc[m][n] = (%SCALAR_DTYPE%)(0);

    // Loop over K dimension, loading A row values and B col values into registers
    for (uint k = 0; k < pc.K; k++) {
        %SCALAR_DTYPE% a[TM];
        %SCALAR_DTYPE% b[TN];

        [unroll] for (uint m = 0; m < TM; m++)
            a[m] = (baseRow + m < pc.M) ? dataA[(baseRow + m) * pc.K + k] : (%SCALAR_DTYPE%)(0);

        [unroll] for (uint n = 0; n < TN; n++)
            b[n] = (baseCol + n < pc.N) ? dataB[k * pc.N + (baseCol + n)] : (%SCALAR_DTYPE%)(0);

        // Outer product: accumulate a * b^T
        [unroll] for (uint m = 0; m < TM; m++)
            [unroll] for (uint n = 0; n < TN; n++)
                acc[m][n] += a[m] * b[n];
    }

    // Write results
    [unroll] for (uint m = 0; m < TM; m++)
        [unroll] for (uint n = 0; n < TN; n++)
            if (baseRow + m < pc.M && baseCol + n < pc.N)
                dataC[(baseRow + m) * pc.N + (baseCol + n)] = acc[m][n];
}
