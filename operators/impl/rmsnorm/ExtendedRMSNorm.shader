#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint dim;          // Actual vector dimension
    uint alignedDim;   // Padded to multiple of 4
    float eps;         // Normalization epsilon (1e-5)
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> residual_base;
[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> delta;
[[vk::binding(2, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> weight;
[[vk::binding(3, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> output;

groupshared float sharedSumSquares[256];
groupshared float sharedResidual[2048];  // Max supported dim (adjust as needed)

[numthreads(256, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 DTid : SV_DispatchThreadID) {
    uint tid = GTid.x;
    uint gid = DTid.x;

    // Phase 1: Compute residual and sum of squares via parallel reduction
    float localSum = 0.0;
    for (uint i = tid; i < pc.dim; i += 256) {
        uint idx = i;
        float base = float(residual_base[idx]);
        float d = float(delta[idx]);
        float res = base + d;

        // Store residual in shared memory for later use
        if (i < 2048) {  // Bounds check for shared memory
            sharedResidual[i] = res;
        }

        localSum += res * res;
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
    // Thread 0 computes: scale = rsqrt(mean(residual²) + eps)
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
        float res;
        if (i < 2048) {
            // Use cached residual from shared memory (faster)
            res = sharedResidual[i];
        } else {
            // Recompute for dimensions > 2048
            res = float(residual_base[idx]) + float(delta[idx]);
        }
        float wVal = float(weight[idx]);
        output[idx] = (%SCALAR_DTYPE_INPUT%)(res * scale * wVal);
    }
}
