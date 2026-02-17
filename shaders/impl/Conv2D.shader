#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
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
    uint dilationH;       // dilation height
    uint dilationW;       // dilation width
    uint groups;          // groups
    uint H_out;           // output height
    uint W_out;           // output width
    uint inAlignedW;      // aligned input innermost stride
    uint outAlignedW;     // aligned output innermost stride
    uint weightAlignedKW; // aligned weight innermost stride
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferInput {
    %SCALAR_DTYPE% input_data[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferWeight {
    %SCALAR_DTYPE% weight_data[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %SCALAR_DTYPE% output_data[];
};

void main() {
    uint w_out = gl_GlobalInvocationID.x;
    uint linear_y = gl_GlobalInvocationID.y;

    // Decode linear_y into (n, c_out, h_out)
    uint h_out = linear_y % H_out;
    uint nc = linear_y / H_out;
    uint c_out = nc % C_out;
    uint n = nc / C_out;

    if (w_out >= W_out || n >= batchSize) return;

    uint C_in_per_group = C_in / groups;
    uint C_out_per_group = C_out / groups;
    uint g = c_out / C_out_per_group;

    %SCALAR_DTYPE% sum = %SCALAR_DTYPE%(0);

    for (uint ci = 0; ci < C_in_per_group; ci++) {
        uint c_in_abs = g * C_in_per_group + ci;

        for (uint kh = 0; kh < kH; kh++) {
            int h_in = int(h_out * strideH + kh * dilationH) - int(padH);
            if (h_in < 0 || h_in >= int(H_in)) continue;

            for (uint kw = 0; kw < kW; kw++) {
                int w_in = int(w_out * strideW + kw * dilationW) - int(padW);
                if (w_in < 0 || w_in >= int(W_in)) continue;

                uint in_idx = n * C_in * H_in * inAlignedW
                            + c_in_abs * H_in * inAlignedW
                            + uint(h_in) * inAlignedW
                            + uint(w_in);

                uint w_idx = c_out * C_in_per_group * kH * weightAlignedKW
                           + ci * kH * weightAlignedKW
                           + kh * weightAlignedKW
                           + kw;

                sum += input_data[in_idx] * weight_data[w_idx];
            }
        }
    }

    uint out_idx = n * C_out * H_out * outAlignedW
                 + c_out * H_out * outAlignedW
                 + h_out * outAlignedW
                 + w_out;

    output_data[out_idx] = sum;
}
