#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Register-tiled matrix multiplication: each thread computes a TM x TN block
// of the output using only registers (local variables). No shared memory.
// Demonstrates register-level data reuse without cooperative loading.

#define TM 4
#define TN 4

#include "MatMulCommon.shaderh"

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint baseRow = DTid.y * TM;
    uint baseCol = DTid.x * TN;

    // Accumulator registers for TM x TN output block
    %SCALAR_DTYPE_OUTPUT% acc[TM][TN];
    [unroll] for (uint m = 0; m < TM; m++)
        [unroll] for (uint n = 0; n < TN; n++)
            acc[m][n] = (%SCALAR_DTYPE_OUTPUT%)(0);

    // Loop over K dimension, loading A row values and B col values into registers
    for (uint k = 0; k < pc.K; k++) {
        %SCALAR_DTYPE_INPUT1% a[TM];
        %SCALAR_DTYPE_INPUT2% b[TN];

        [unroll] for (uint m = 0; m < TM; m++)
            a[m] = loadA(baseRow + m, k);

        [unroll] for (uint n = 0; n < TN; n++)
            b[n] = loadB(k, baseCol + n);

        // Outer product: accumulate a * b^T
        [unroll] for (uint m = 0; m < TM; m++)
            [unroll] for (uint n = 0; n < TN; n++)
                acc[m][n] += (%SCALAR_DTYPE_OUTPUT%)(a[m]) * (%SCALAR_DTYPE_OUTPUT%)(b[n]);
    }

    // Write results
    [unroll] for (uint m = 0; m < TM; m++) {
        [unroll] for (uint n = 0; n < TN; n++) {
            if (baseRow + m < pc.M && baseCol + n < pc.N) {
                writeOutput(baseRow + m, baseCol + n, acc[m][n]);
            }
        }
    }
}
