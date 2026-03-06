#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_SCALES%
%DTYPE_DEFINES_OUTPUT%

// Fused Q4 GEMV + SiLU: silu(A * dequant(B)), optimized for M=1.
// Each thread computes one output element with inline SiLU activation.
// where silu(x) = x / (1 + exp(-x))

#include "MatMulQ4Common.shaderh"

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint n = DTid.x;
    uint m = DTid.y;
    if (n >= pc.N || m >= pc.M) return;

    float acc = 0.0f;

    // K-loop unrolled by 4 with mad() for ILP
    uint K4 = pc.K & ~3u;
    uint k = 0;
    for (; k < K4; k += 4) {
        acc = mad(float(loadA(m, k)),     loadB(k,     n), acc);
        acc = mad(float(loadA(m, k + 1)), loadB(k + 1, n), acc);
        acc = mad(float(loadA(m, k + 2)), loadB(k + 2, n), acc);
        acc = mad(float(loadA(m, k + 3)), loadB(k + 3, n), acc);
    }
    // Remaining elements
    for (; k < pc.K; k++) {
        acc = mad(float(loadA(m, k)), loadB(k, n), acc);
    }

    // Apply SiLU activation: silu(x) = x / (1 + exp(-x))
    dataC[m * pc.strideC + n] = acc / (1.0f + exp(-acc));
}
