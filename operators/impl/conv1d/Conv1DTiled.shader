#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Tiled Conv1D: loads input segment + halo into shared memory
// Dispatch: x = tiles along L_out, y = N * C_out
// TILE_SIZE=%TILE_SIZE%

#define TILE_SIZE %TILE_SIZE%
#define MAX_KL 33

#include "Conv1DCommon.shaderh"

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> input_data;

[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> weight_data;

[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> output_data;

groupshared %SCALAR_DTYPE_INPUT% sharedInput[TILE_SIZE + MAX_KL - 1];

[numthreads(TILE_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint L_out = (pc.L_in + 2 * pc.padding - pc.kL) / pc.stride + 1;
    uint inAlignedL = (pc.L_in + 3) & ~3u;
    uint outAlignedL = (L_out + 3) & ~3u;
    uint weightAlignedKL = (pc.kL + 3) & ~3u;

    // x tiles over L_out, y encodes (n, c_out)
    uint l_out = DTid.x;
    uint c_out = DTid.y % pc.C_out;
    uint n = DTid.y / pc.C_out;

    bool active = (l_out < L_out);

    %SCALAR_DTYPE_INPUT% sum = (%SCALAR_DTYPE_INPUT%)(0);

    // Process one input channel at a time using shared memory
    for (uint ci = 0; ci < pc.C_in; ci++) {
        uint sharedLen = min(TILE_SIZE + pc.kL - 1, TILE_SIZE + MAX_KL - 1);
        int baseL = int(Gid.x * TILE_SIZE * pc.stride) - int(pc.padding);

        // Cooperatively load input segment + halo into shared memory
        for (uint s = GTid.x; s < sharedLen; s += TILE_SIZE) {
            int il = baseL + int(s);
            %SCALAR_DTYPE_INPUT% val = (%SCALAR_DTYPE_INPUT%)(0);
            if (il >= 0 && il < int(pc.L_in)) {
                uint in_idx = n * pc.C_in * inAlignedL + ci * inAlignedL + uint(il);
                val = input_data[in_idx];
            }
            sharedInput[s] = val;
        }
        GroupMemoryBarrierWithGroupSync();

        // Compute convolution from shared memory
        if (active) {
            uint localL = GTid.x * pc.stride;
            for (uint kl = 0; kl < pc.kL; kl++) {
                if (localL + kl < sharedLen) {
                    uint w_idx = c_out * pc.C_in * weightAlignedKL + ci * weightAlignedKL + kl;
                    sum += sharedInput[localL + kl] * weight_data[w_idx];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (active) {
        uint out_idx = n * pc.C_out * outAlignedL + c_out * outAlignedL + l_out;
        output_data[out_idx] = sum;
    }
}
