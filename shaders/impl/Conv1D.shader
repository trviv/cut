#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint batchSize;       // N
    uint C_in;            // input channels
    uint L_in;            // input length
    uint C_out;           // output channels
    uint kL;              // kernel length
    uint stride;          // stride
    uint padding;         // padding
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
    uint gid = gl_GlobalInvocationID.x;

    uint L_out = (L_in + 2 * padding - kL) / stride + 1;
    uint inAlignedL = (L_in + 3) & ~3u;
    uint outAlignedL = (L_out + 3) & ~3u;
    uint weightAlignedKL = (kL + 3) & ~3u;

    // Decode linear index into (n, c_out, l_out)
    uint l_out = gid % L_out;
    uint nc = gid / L_out;
    uint c_out = nc % C_out;
    uint n = nc / C_out;

    if (n >= batchSize) return;

    %SCALAR_DTYPE% sum = %SCALAR_DTYPE%(0);

    for (uint ci = 0; ci < C_in; ci++) {
        for (uint kl = 0; kl < kL; kl++) {
            int l_in = int(l_out * stride + kl) - int(padding);
            if (l_in < 0 || l_in >= int(L_in)) continue;

            uint in_idx = n * C_in * inAlignedL
                        + ci * inAlignedL
                        + uint(l_in);

            uint w_idx = c_out * C_in * weightAlignedKL
                       + ci * weightAlignedKL
                       + kl;

            sum += input_data[in_idx] * weight_data[w_idx];
        }
    }

    uint out_idx = n * C_out * outAlignedL
                 + c_out * outAlignedL
                 + l_out;

    output_data[out_idx] = sum;
}
