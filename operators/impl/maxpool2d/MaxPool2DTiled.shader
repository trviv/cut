#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Tiled MaxPool2D: loads input tile + halo into shared memory
// Dispatch: x = tiles along W_out, y = tiles along H_out, z = N * C
// TILE_W=%TILE_W%, TILE_H=%TILE_H%

#define TILE_W %TILE_W%
#define TILE_H %TILE_H%
#define MAX_KH 11
#define MAX_KW 11

// Shared memory dimensions (support stride up to 2)
#define SHARED_H (TILE_H * 2 + MAX_KH - 1)
#define SHARED_W (TILE_W * 2 + MAX_KW - 1)

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

groupshared %SCALAR_DTYPE% sharedInput[SHARED_H][SHARED_W];

[numthreads(TILE_W, TILE_H, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint H_out = (pc.H_in + 2 * pc.padH - pc.kernelH) / pc.strideH + 1;
    uint W_out = (pc.W_in + 2 * pc.padW - pc.kernelW) / pc.strideW + 1;
    uint inAlignedW = (pc.W_in + 3) & ~3u;
    uint outAlignedW = (W_out + 3) & ~3u;

    // z encodes (n, c), x/y are spatial
    uint c = DTid.z % pc.C;
    uint n = DTid.z / pc.C;
    uint h_out = DTid.y;
    uint w_out = DTid.x;

    bool active = (w_out < W_out && h_out < H_out);

    // Actual shared memory region needed for this tile
    uint sharedH = (TILE_H - 1) * pc.strideH + pc.kernelH;
    uint sharedW = (TILE_W - 1) * pc.strideW + pc.kernelW;

    int baseH = int(Gid.y * TILE_H * pc.strideH) - int(pc.padH);
    int baseW = int(Gid.x * TILE_W * pc.strideW) - int(pc.padW);

    // Cooperatively load input tile + halo into shared memory
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
            if (sh < SHARED_H && sw < SHARED_W) {
                sharedInput[sh][sw] = val;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (!active) return;

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
