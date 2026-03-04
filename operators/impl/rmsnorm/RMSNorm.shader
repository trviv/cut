#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint dim;          // Actual vector dimension
    uint alignedDim;   // Padded to multiple of 4
    float eps;         // Normalization epsilon (1e-5)
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> x;
[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> weight;
[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> output;

groupshared float sharedSumSquares[256];

[numthreads(256, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 DTid : SV_DispatchThreadID) {
    uint tid = GTid.x;
    uint gid = DTid.x;

    // Phase 1: Compute sum of squares via parallel reduction
    float localSum = 0.0;
    for (uint i = tid; i < pc.dim; i += 256) {
        // Use actual dimension for iteration, aligned dimension for indexing
        uint idx = i;  // Assuming 1D data, no row/col calculation needed
        float val = float(x[idx]);
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
    // Thread 0 computes: scale = rsqrt(mean(x²) + eps) = rsqrt(sum/dim + eps)
    float scale;
    if (tid == 0) {
        float meanSquare = sharedSumSquares[0] / float(pc.dim);
        scale = rsqrt(meanSquare + pc.eps);
        sharedSumSquares[0] = scale;  // Store scale for other threads
    }
    GroupMemoryBarrierWithGroupSync();
    scale = sharedSumSquares[0];  // All threads read the scale

    // Phase 4: Apply normalization and weight
    for (uint i = tid; i < pc.dim; i += 256) {
        uint idx = i;
        float xVal = float(x[idx]);
        float wVal = float(weight[idx]);
        output[idx] = (%SCALAR_DTYPE_INPUT%)(xVal * scale * wVal);
    }
}
