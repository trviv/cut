#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// GEMV: matrix-vector multiply optimized for M=1 (autoregressive decoding).
// C[m, N] = A[m, K] * B[K, N]
//
// Optimization: loads input vector A into shared memory once per workgroup,
// eliminating redundant global memory reads across threads. Each thread
// accumulates over K for its assigned output column, reading A from shared
// memory (fast) and B from global memory with coalesced access.

#include "MatMulCommon.shaderh"

#define WG_SIZE 256

groupshared %SCALAR_DTYPE_INPUT1% sharedA[2048]; // K up to 2048

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID,
          uint3 DTid : SV_DispatchThreadID) {
    uint n = DTid.x;
    uint m = DTid.y;
    uint tid = GTid.x;

    // Cooperatively load A[m, 0..K-1] into shared memory
    for (uint i = tid; i < pc.K; i += WG_SIZE) {
        sharedA[i] = loadA_fast(m, i);
    }
    GroupMemoryBarrierWithGroupSync();

    if (n >= pc.N || m >= pc.M) return;

    %SCALAR_DTYPE_OUTPUT% acc = (%SCALAR_DTYPE_OUTPUT%)(0);

    // K-loop: A from shared memory (broadcast), B from global (coalesced)
    uint K4 = pc.K & ~3u;
    uint k = 0;
    for (; k < K4; k += 4) {
        acc = mad((%SCALAR_DTYPE_OUTPUT%)sharedA[k],
                  (%SCALAR_DTYPE_OUTPUT%)loadB_fast(k, n), acc);
        acc = mad((%SCALAR_DTYPE_OUTPUT%)sharedA[k + 1],
                  (%SCALAR_DTYPE_OUTPUT%)loadB_fast(k + 1, n), acc);
        acc = mad((%SCALAR_DTYPE_OUTPUT%)sharedA[k + 2],
                  (%SCALAR_DTYPE_OUTPUT%)loadB_fast(k + 2, n), acc);
        acc = mad((%SCALAR_DTYPE_OUTPUT%)sharedA[k + 3],
                  (%SCALAR_DTYPE_OUTPUT%)loadB_fast(k + 3, n), acc);
    }
    for (; k < pc.K; k++) {
        acc = mad((%SCALAR_DTYPE_OUTPUT%)sharedA[k],
                  (%SCALAR_DTYPE_OUTPUT%)loadB_fast(k, n), acc);
    }

    writeOutput(m, n, acc);
}
