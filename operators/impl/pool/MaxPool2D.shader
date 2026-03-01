#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint N;
    uint C;
    uint H_in;
    uint W_in;
    uint kernelH;
    uint kernelW;
    uint strideH;
    uint strideW;
    uint padH;
    uint padW;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> input_data;

[[vk::binding(1, 0)]] RWStructuredBuffer<%VEC_DTYPE_INPUT%> output_data;

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint w_out4 = DTid.x;
    uint linear_y = DTid.y;

    uint H_out = (pc.H_in + 2 * pc.padH - pc.kernelH) / pc.strideH + 1;
    uint W_out = (pc.W_in + 2 * pc.padW - pc.kernelW) / pc.strideW + 1;
    uint inAlignedW = (pc.W_in + 3) & ~3u;
    uint outAlignedW4 = ((W_out + 3) & ~3u) / 4;

    // Decode linear_y into (n, c, h_out)
    uint h_out = linear_y % H_out;
    uint nc = linear_y / H_out;
    uint c = nc % pc.C;
    uint n = nc / pc.C;

    if (w_out4 >= outAlignedW4 || n >= pc.N) return;

    uint baseW = w_out4 * 4;
    %VEC_DTYPE_INPUT% maxVals = (%VEC_DTYPE_INPUT%)((%SCALAR_DTYPE_INPUT%)(-1.0e38));

    for (uint kh = 0; kh < pc.kernelH; kh++) {
        int h_in = int(h_out * pc.strideH + kh) - int(pc.padH);
        if (h_in < 0 || h_in >= int(pc.H_in)) continue;

        for (uint kw = 0; kw < pc.kernelW; kw++) {
            for (uint i = 0; i < 4; i++) {
                uint w_out = baseW + i;
                if (w_out >= W_out) continue;
                int w_in = int(w_out * pc.strideW + kw) - int(pc.padW);
                if (w_in < 0 || w_in >= int(pc.W_in)) continue;

                uint in_idx = n * pc.C * pc.H_in * inAlignedW
                            + c * pc.H_in * inAlignedW
                            + uint(h_in) * inAlignedW
                            + uint(w_in);

                maxVals[i] = max(maxVals[i], input_data[in_idx]);
            }
        }
    }

    // Zero out padding positions
    for (uint i = 0; i < 4; i++) {
        if (baseW + i >= W_out) maxVals[i] = (%SCALAR_DTYPE_INPUT%)(0);
    }

    uint out_idx = n * pc.C * H_out * outAlignedW4
                 + c * H_out * outAlignedW4
                 + h_out * outAlignedW4
                 + w_out4;

    output_data[out_idx] = maxVals;
}
