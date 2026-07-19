// Native CUDA counterpart of Conv2DTiled.shader (block dims TILE_W x TILE_H x 1;
// shared-memory input tile + halo) — keep semantics in lockstep.
#include "Conv2DCommon.cuh"

#ifndef TILE_W
#define TILE_W 16
#endif
#ifndef TILE_H
#define TILE_H 16
#endif
#define MAX_KH 11
#define MAX_KW 11
#define SHARED_H (TILE_H * 2 + MAX_KH - 1)
#define SHARED_W (TILE_W * 2 + MAX_KW - 1)

extern "C" __global__ void cut_main(const cut_conv2d_t* __restrict__ input_data,
                                    const cut_conv2d_t* __restrict__ weight_data,
                                    cut_conv2d_t* __restrict__ output_data,
                                    PushConstants pc) {
    __shared__ cut_conv2d_t sharedInput[SHARED_H][SHARED_W];

    uint H_out = (pc.H_in + 2 * pc.padH - pc.kH) / pc.strideH + 1;
    uint W_out = (pc.W_in + 2 * pc.padW - pc.kW) / pc.strideW + 1;
    uint inAlignedW = (pc.W_in + 3) & ~3u;
    uint outAlignedW = (W_out + 3) & ~3u;
    uint weightAlignedKW = (pc.kW + 3) & ~3u;

    // z encodes (n, c_out), x/y are spatial
    uint dz = blockIdx.z * blockDim.z + threadIdx.z;
    uint c_out = dz % pc.C_out;
    uint n = dz / pc.C_out;
    uint h_out = blockIdx.y * TILE_H + threadIdx.y;
    uint w_out = blockIdx.x * TILE_W + threadIdx.x;

    bool active = (w_out < W_out && h_out < H_out);

    cut_conv2d_t sum = (cut_conv2d_t)(0);

    // Process one input channel at a time using shared memory
    for (uint ci = 0; ci < pc.C_in; ci++) {
        // Actual shared memory region needed for this tile
        uint sharedH = (TILE_H - 1) * pc.strideH + pc.kH;
        uint sharedW = (TILE_W - 1) * pc.strideW + pc.kW;

        // Base input position for this tile
        int baseH = int(blockIdx.y * TILE_H * pc.strideH) - int(pc.padH);
        int baseW = int(blockIdx.x * TILE_W * pc.strideW) - int(pc.padW);

        // Cooperatively load input tile + halo into shared memory
        for (uint sh = threadIdx.y; sh < sharedH; sh += TILE_H) {
            for (uint sw = threadIdx.x; sw < sharedW; sw += TILE_W) {
                int ih = baseH + int(sh);
                int iw = baseW + int(sw);
                cut_conv2d_t val = (cut_conv2d_t)(0);
                if (ih >= 0 && ih < int(pc.H_in) && iw >= 0 && iw < int(pc.W_in)) {
                    uint in_idx = n * pc.C_in * pc.H_in * inAlignedW
                                + ci * pc.H_in * inAlignedW
                                + uint(ih) * inAlignedW
                                + uint(iw);
                    val = input_data[in_idx];
                }
                if (sh < SHARED_H && sw < SHARED_W) {
                    sharedInput[sh][sw] = val;
                }
            }
        }
        __syncthreads();

        // Compute convolution from shared memory
        if (active) {
            uint localH = threadIdx.y * pc.strideH;
            uint localW = threadIdx.x * pc.strideW;

            for (uint kh = 0; kh < pc.kH; kh++) {
                for (uint kw = 0; kw < pc.kW; kw++) {
                    uint w_idx = c_out * pc.C_in * pc.kH * weightAlignedKW
                               + ci * pc.kH * weightAlignedKW
                               + kh * weightAlignedKW
                               + kw;
                    sum += sharedInput[localH + kh][localW + kw] * weight_data[w_idx];
                }
            }
        }

        __syncthreads();
    }

    if (active) {
        uint out_idx = n * pc.C_out * H_out * outAlignedW
                     + c_out * H_out * outAlignedW
                     + h_out * outAlignedW
                     + w_out;

        output_data[out_idx] = sum;
    }
}
