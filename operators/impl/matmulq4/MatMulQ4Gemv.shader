#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_SCALES%
%DTYPE_DEFINES_OUTPUT%

// Q4 GEMV: matrix-vector multiply optimized for M=1 (autoregressive decoding).
// Each thread computes one output element: C[m, n] = sum_k( A[m,k] * dequant(B[k,n]) )
// Adjacent threads access adjacent B columns — coalesced memory reads.
// Supports arbitrary M via dispatch Y dimension (DTid.y = row index).

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

    dataC[m * pc.strideC + n] = acc;
}
