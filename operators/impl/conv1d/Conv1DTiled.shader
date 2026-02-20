#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Tiled Conv1D: loads input segment + halo into shared memory
// TILE_SIZE=%TILE_SIZE%

#define TILE_SIZE %TILE_SIZE%
#define MAX_KL 33

struct PushConstants {
    uint batchSize;       // N
    uint C_in;            // input channels
    uint L_in;            // input length
    uint C_out;           // output channels
    uint kL;              // kernel length
    uint stride;          // stride
    uint padding;         // padding
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> input_data;

[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE%> weight_data;

[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> output_data;

groupshared %SCALAR_DTYPE% sharedInput[TILE_SIZE + MAX_KL - 1];

[numthreads(TILE_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint gid = DTid.x;

    uint L_out = (pc.L_in + 2 * pc.padding - pc.kL) / pc.stride + 1;
    uint inAlignedL = (pc.L_in + 3) & ~3u;
    uint outAlignedL = (L_out + 3) & ~3u;
    uint weightAlignedKL = (pc.kL + 3) & ~3u;

    // Decode linear index into (n, c_out, l_out)
    uint l_out = gid % L_out;
    uint nc = gid / L_out;
    uint c_out = nc % pc.C_out;
    uint n = nc / pc.C_out;

    if (n >= pc.batchSize) return;

    %SCALAR_DTYPE% sum = (%SCALAR_DTYPE%)(0);

    // Process one input channel at a time using shared memory
    for (uint ci = 0; ci < pc.C_in; ci++) {
        uint sharedLen = TILE_SIZE * pc.stride + pc.kL - 1;
        int baseL = int(Gid.x * TILE_SIZE * pc.stride) - int(pc.padding);

        // Cooperatively load input segment + halo into shared memory
        for (uint s = GTid.x; s < sharedLen; s += TILE_SIZE) {
            int il = baseL + int(s);
            %SCALAR_DTYPE% val = (%SCALAR_DTYPE%)(0);
            if (il >= 0 && il < int(pc.L_in)) {
                uint in_idx = n * pc.C_in * inAlignedL + ci * inAlignedL + uint(il);
                val = input_data[in_idx];
            }
            if (s < TILE_SIZE + MAX_KL - 1) {
                sharedInput[s] = val;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Compute convolution from shared memory
        uint localL = GTid.x * pc.stride;
        for (uint kl = 0; kl < pc.kL; kl++) {
            uint w_idx = c_out * pc.C_in * weightAlignedKL + ci * weightAlignedKL + kl;
            if (localL + kl < TILE_SIZE + MAX_KL - 1) {
                sum += sharedInput[localL + kl] * weight_data[w_idx];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (gid < pc.batchSize * pc.C_out * L_out) {
        uint out_idx = n * pc.C_out * outAlignedL + c_out * outAlignedL + l_out;
        output_data[out_idx] = sum;
    }
}
