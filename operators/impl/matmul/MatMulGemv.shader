#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// GEMV: matrix-vector multiply optimized for M=1 (autoregressive decoding).
// Each thread computes one output element: C[m, n] = dot(A[m, :], B[:, n])
// Adjacent threads access adjacent B columns — coalesced memory reads.
// Supports arbitrary M via dispatch Y dimension (DTid.y = row index).

#include "MatMulCommon.shaderh"

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint n = DTid.x;
    uint m = DTid.y;
    if (n >= pc.N || m >= pc.M) return;

    %SCALAR_DTYPE_OUTPUT% acc = (%SCALAR_DTYPE_OUTPUT%)(0);

    // K-loop unrolled by 4 with mad() for ILP
    uint K4 = pc.K & ~3u;
    uint k = 0;
    for (; k < K4; k += 4) {
        acc = mad((%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k),
                  (%SCALAR_DTYPE_OUTPUT%)loadB_fast(k, n), acc);
        acc = mad((%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k + 1),
                  (%SCALAR_DTYPE_OUTPUT%)loadB_fast(k + 1, n), acc);
        acc = mad((%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k + 2),
                  (%SCALAR_DTYPE_OUTPUT%)loadB_fast(k + 2, n), acc);
        acc = mad((%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k + 3),
                  (%SCALAR_DTYPE_OUTPUT%)loadB_fast(k + 3, n), acc);
    }
    // Remaining elements
    for (; k < pc.K; k++) {
        acc = mad((%SCALAR_DTYPE_OUTPUT%)loadA_fast(m, k),
                  (%SCALAR_DTYPE_OUTPUT%)loadB_fast(k, n), acc);
    }

    dataC[m * pc.strideB + n] = acc;
}
