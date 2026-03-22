#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint dim;          // Actual vector dimension (innermost)
    uint alignedDim;   // Padded to multiple of 4 (row stride for 2D)
    float eps;         // Normalization epsilon (1e-5)
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> x;
[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> weight;
[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> output;

groupshared float sharedSumSquares[256];

// Dispatch: (256, batchSize, 1).  Each Y-workgroup normalizes one row.
// For 1D input (batchSize=1), Gid.y=0 and rowOff=0 — same as before.
[numthreads(256, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;
    uint row = Gid.y;                     // Row index (0 for 1D)
    uint rowOff = row * pc.alignedDim;    // Byte-aligned row offset

    // Phase 1: Compute sum of squares for this row
    float localSum = 0.0;
    for (uint i = tid; i < pc.dim; i += 256) {
        float val = float(x[rowOff + i]);
        localSum += val * val;
    }
    sharedSumSquares[tid] = localSum;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Tree reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSumSquares[tid] += sharedSumSquares[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Phase 3: Compute normalization scale
    float scale;
    if (tid == 0) {
        float meanSquare = sharedSumSquares[0] / float(pc.dim);
        scale = rsqrt(meanSquare + pc.eps);
        sharedSumSquares[0] = scale;
    }
    GroupMemoryBarrierWithGroupSync();
    scale = sharedSumSquares[0];

    // Phase 4: Apply normalization and weight (weight is shared across rows)
    for (uint i = tid; i < pc.dim; i += 256) {
        float xVal = float(x[rowOff + i]);
        float wVal = float(weight[i]);
        output[rowOff + i] = (%SCALAR_DTYPE_INPUT%)(xVal * scale * wVal);
    }
}
