#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Tiled MaxPool2D: loads input tile + halo into shared memory
// TILE_W=%TILE_W%, TILE_H=%TILE_H%

#define TILE_W %TILE_W%
#define TILE_H %TILE_H%
#define MAX_KH 11
#define MAX_KW 11

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

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> input_data;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> output_data;

groupshared %SCALAR_DTYPE% sharedInput[TILE_H + MAX_KH - 1][TILE_W + MAX_KW - 1];

[numthreads(TILE_W, TILE_H, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint H_out = (pc.H_in + 2 * pc.padH - pc.kernelH) / pc.strideH + 1;
    uint W_out = (pc.W_in + 2 * pc.padW - pc.kernelW) / pc.strideW + 1;
    uint inAlignedW = (pc.W_in + 3) & ~3u;
    uint outAlignedW = (W_out + 3) & ~3u;

    // Decode tile position
    uint w_out = Gid.x * TILE_W + GTid.x;
    uint linear_y = DTid.y;

    uint h_out = linear_y % H_out;
    uint nc = linear_y / H_out;
    uint c = nc % pc.C;
    uint n = nc / pc.C;

    if (n >= pc.N) return;

    // Cooperatively load input tile + halo into shared memory
    uint sharedH = pc.kernelH + TILE_H - 1;
    uint sharedW = pc.kernelW + TILE_W - 1;

    int baseH = int(Gid.y * TILE_H * pc.strideH) - int(pc.padH);
    int baseW = int(Gid.x * TILE_W * pc.strideW) - int(pc.padW);

    // Use the decode from linear_y for the correct (n, c) for loading
    // But we need to handle this carefully since linear_y encodes (n, c, h_out)

    for (uint sh = GTid.y; sh < sharedH; sh += TILE_H) {
        for (uint sw = GTid.x; sw < sharedW; sw += TILE_W) {
            int ih = baseH + int(sh);
            int iw = baseW + int(sw);
#ifdef DTYPE_IS_FLOAT
            %SCALAR_DTYPE% val = (%SCALAR_DTYPE%)(-1.0e38);
#else
            %SCALAR_DTYPE% val = (%SCALAR_DTYPE%)(-2147483648);
#endif
            if (ih >= 0 && ih < int(pc.H_in) && iw >= 0 && iw < int(pc.W_in)) {
                uint in_idx = n * pc.C * pc.H_in * inAlignedW
                            + c * pc.H_in * inAlignedW
                            + uint(ih) * inAlignedW
                            + uint(iw);
                val = input_data[in_idx];
            }
            sharedInput[sh][sw] = val;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (w_out >= W_out) return;

    // Compute max pooling from shared memory
    uint localH = GTid.y * pc.strideH;
    uint localW = GTid.x * pc.strideW;

#ifdef DTYPE_IS_FLOAT
    %SCALAR_DTYPE% maxVal = (%SCALAR_DTYPE%)(-1.0e38);
#else
    %SCALAR_DTYPE% maxVal = (%SCALAR_DTYPE%)(-2147483648);
#endif

    for (uint kh = 0; kh < pc.kernelH; kh++) {
        for (uint kw = 0; kw < pc.kernelW; kw++) {
            maxVal = max(maxVal, sharedInput[localH + kh][localW + kw]);
        }
    }

    uint out_idx = n * pc.C * H_out * outAlignedW
                 + c * H_out * outAlignedW
                 + h_out * outAlignedW
                 + w_out;

    output_data[out_idx] = maxVal;
}
