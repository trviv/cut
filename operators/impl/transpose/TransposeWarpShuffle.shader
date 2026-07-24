#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// HLSL counterpart of TransposeWarpShuffle.cu. The CUDA side does a warp-register
// 32x32 transpose via __shfl_xor_sync; wave/subgroup size is device-dependent in
// Vulkan (32 on NVIDIA, other values on llvmpipe/AMD), so this reference path
// uses a subgroup-size-AGNOSTIC shared-memory transpose instead. Both produce
// identical output for the same variant. Block = [32,1,1] (one 32-lane group),
// one 32x32 tile per group, matching the .cu's launch geometry and push
// constants exactly.

#define WARP 32

struct PushConstants {
    uint M;
    uint N;
    uint strideIn;
    uint strideOut;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> dataOut;

// +1 column padding avoids bank conflicts on the transposed shared read.
groupshared %SCALAR_DTYPE_INPUT% tile[WARP][WARP + 1];

[numthreads(WARP, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tileX = Gid.x;
    uint tileY = Gid.y;

    uint l = GTid.x; // 0..31
    uint rowBase = tileY * WARP; // input row base (M)
    uint colBase = tileX * WARP; // input col base (N)

    // Load column l into shared: tile[r][l] = A[r][l]. Coalesced over l.
    [unroll] for (uint r = 0; r < WARP; r++) {
        uint inRow = rowBase + r;
        uint inCol = colBase + l;
        tile[r][l] = (inRow < pc.M && inCol < pc.N)
                         ? dataIn[inRow * pc.strideIn + inCol]
                         : (%SCALAR_DTYPE_INPUT%)(0);
    }

    GroupMemoryBarrierWithGroupSync();

    // Write transposed: out[colBase+r][rowBase+l] = tile[l][r] = A[l][r].
    // Coalesced over l; transposed shared read hits distinct banks (stride 33).
    [unroll] for (uint r = 0; r < WARP; r++) {
        uint outRow = colBase + r;
        uint outCol = rowBase + l;
        if (outRow < pc.N && outCol < pc.M) {
            dataOut[outRow * pc.strideOut + outCol] = tile[l][r];
        }
    }
}
