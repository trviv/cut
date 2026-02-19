#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

struct PushConstants {
    uint batchSize;       // N
    uint C_in;            // input channels
    uint H_in;            // input height
    uint W_in;            // input width
    uint C_out;           // output channels
    uint kH;              // kernel height
    uint kW;              // kernel width
    uint strideH;         // stride height
    uint strideW;         // stride width
    uint padH;            // padding height
    uint padW;            // padding width
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> input_data;

[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE%> weight_data;

[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> output_data;

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint w_out = DTid.x;
    uint linear_y = DTid.y;

    uint H_out = (pc.H_in + 2 * pc.padH - pc.kH) / pc.strideH + 1;
    uint W_out = (pc.W_in + 2 * pc.padW - pc.kW) / pc.strideW + 1;
    uint inAlignedW = (pc.W_in + 3) & ~3u;
    uint outAlignedW = (W_out + 3) & ~3u;
    uint weightAlignedKW = (pc.kW + 3) & ~3u;

    // Decode linear_y into (n, c_out, h_out)
    uint h_out = linear_y % H_out;
    uint nc = linear_y / H_out;
    uint c_out = nc % pc.C_out;
    uint n = nc / pc.C_out;

    if (w_out >= W_out || n >= pc.batchSize) return;

    %SCALAR_DTYPE% sum = (%SCALAR_DTYPE%)(0);

    for (uint ci = 0; ci < pc.C_in; ci++) {
        for (uint kh = 0; kh < pc.kH; kh++) {
            int h_in = int(h_out * pc.strideH + kh) - int(pc.padH);
            if (h_in < 0 || h_in >= int(pc.H_in)) continue;

            for (uint kw = 0; kw < pc.kW; kw++) {
                int w_in = int(w_out * pc.strideW + kw) - int(pc.padW);
                if (w_in < 0 || w_in >= int(pc.W_in)) continue;

                uint in_idx = n * pc.C_in * pc.H_in * inAlignedW
                            + ci * pc.H_in * inAlignedW
                            + uint(h_in) * inAlignedW
                            + uint(w_in);

                uint w_idx = c_out * pc.C_in * pc.kH * weightAlignedKW
                           + ci * pc.kH * weightAlignedKW
                           + kh * weightAlignedKW
                           + kw;

                sum += input_data[in_idx] * weight_data[w_idx];
            }
        }
    }

    uint out_idx = n * pc.C_out * H_out * outAlignedW
                 + c_out * H_out * outAlignedW
                 + h_out * outAlignedW
                 + w_out;

    output_data[out_idx] = sum;
}
