#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Tiled Conv2D: loads input tile + halo into shared memory
// TILE_W=%TILE_W%, TILE_H=%TILE_H%

#define TILE_W %TILE_W%
#define TILE_H %TILE_H%

// Max kernel size supported in shared memory (halo region)
#define MAX_KH 11
#define MAX_KW 11

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

// Shared memory for input tile with halo
groupshared %SCALAR_DTYPE% sharedInput[TILE_H + MAX_KH - 1][TILE_W + MAX_KW - 1];

[numthreads(TILE_W, TILE_H, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
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

    // Process one input channel at a time using shared memory
    for (uint ci = 0; ci < pc.C_in; ci++) {
        // Cooperatively load input tile + halo into shared memory
        uint sharedH = pc.kH + TILE_H - 1;
        uint sharedW = pc.kW + TILE_W - 1;

        // Base input position for this tile
        int baseH = int(Gid.y * TILE_H * pc.strideH) - int(pc.padH);
        int baseW = int(Gid.x * TILE_W * pc.strideW) - int(pc.padW);

        // Each thread loads multiple elements to fill shared memory
        for (uint sh = GTid.y; sh < sharedH; sh += TILE_H) {
            for (uint sw = GTid.x; sw < sharedW; sw += TILE_W) {
                int ih = baseH + int(sh);
                int iw = baseW + int(sw);
                %SCALAR_DTYPE% val = (%SCALAR_DTYPE%)(0);
                if (ih >= 0 && ih < int(pc.H_in) && iw >= 0 && iw < int(pc.W_in)) {
                    uint in_idx = n * pc.C_in * pc.H_in * inAlignedW
                                + ci * pc.H_in * inAlignedW
                                + uint(ih) * inAlignedW
                                + uint(iw);
                    val = input_data[in_idx];
                }
                sharedInput[sh][sw] = val;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Compute convolution from shared memory
        uint localH = GTid.y * pc.strideH;
        uint localW = GTid.x * pc.strideW;

        for (uint kh = 0; kh < pc.kH; kh++) {
            for (uint kw = 0; kw < pc.kW; kw++) {
                uint w_idx = c_out * pc.C_in * pc.kH * weightAlignedKW
                           + ci * pc.kH * weightAlignedKW
                           + kh * weightAlignedKW
                           + kw;
                sum += sharedInput[localH + kh][localW + kw] * weight_data[w_idx];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    uint out_idx = n * pc.C_out * H_out * outAlignedW
                 + c_out * H_out * outAlignedW
                 + h_out * outAlignedW
                 + w_out;

    output_data[out_idx] = sum;
}
