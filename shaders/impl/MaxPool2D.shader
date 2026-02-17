#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
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

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferInput {
    %SCALAR_DTYPE% input_data[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %SCALAR_DTYPE% output_data[];
};

void main() {
    uint w_out = gl_GlobalInvocationID.x;
    uint linear_y = gl_GlobalInvocationID.y;

    uint H_out = (H_in + 2 * padH - kernelH) / strideH + 1;
    uint W_out = (W_in + 2 * padW - kernelW) / strideW + 1;
    uint inAlignedW = (W_in + 3) & ~3u;
    uint outAlignedW = (W_out + 3) & ~3u;

    // Decode linear_y into (n, c, h_out)
    uint h_out = linear_y % H_out;
    uint nc = linear_y / H_out;
    uint c = nc % C;
    uint n = nc / C;

    if (w_out >= W_out || n >= N) return;

    %SCALAR_DTYPE% maxVal = %SCALAR_DTYPE%(-1.0e38);

    for (uint kh = 0; kh < kernelH; kh++) {
        int h_in = int(h_out * strideH + kh) - int(padH);
        if (h_in < 0 || h_in >= int(H_in)) continue;

        for (uint kw = 0; kw < kernelW; kw++) {
            int w_in = int(w_out * strideW + kw) - int(padW);
            if (w_in < 0 || w_in >= int(W_in)) continue;

            uint in_idx = n * C * H_in * inAlignedW
                        + c * H_in * inAlignedW
                        + uint(h_in) * inAlignedW
                        + uint(w_in);

            maxVal = max(maxVal, input_data[in_idx]);
        }
    }

    uint out_idx = n * C * H_out * outAlignedW
                 + c * H_out * outAlignedW
                 + h_out * outAlignedW
                 + w_out;

    output_data[out_idx] = maxVal;
}
