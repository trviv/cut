#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#include "Conv1DCommon.shaderh"

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> input_data;

[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> weight_data;

[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> output_data;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
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

    %SCALAR_DTYPE_INPUT% sum = (%SCALAR_DTYPE_INPUT%)(0);

    for (uint ci = 0; ci < pc.C_in; ci++) {
        for (uint kl = 0; kl < pc.kL; kl++) {
            int l_in = int(l_out * pc.stride + kl) - int(pc.padding);
            if (l_in < 0 || l_in >= int(pc.L_in)) continue;

            uint in_idx = n * pc.C_in * inAlignedL
                        + ci * inAlignedL
                        + uint(l_in);

            uint w_idx = c_out * pc.C_in * weightAlignedKL
                       + ci * weightAlignedKL
                       + kl;

            sum += input_data[in_idx] * weight_data[w_idx];
        }
    }

    uint out_idx = n * pc.C_out * outAlignedL
                 + c_out * outAlignedL
                 + l_out;

    output_data[out_idx] = sum;
}
