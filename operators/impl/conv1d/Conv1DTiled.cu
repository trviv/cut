// Native CUDA counterpart of Conv1DTiled.shader (block dims TILE_SIZE x 1 x 1;
// shared-memory input tile + halo) — keep semantics in lockstep.
#include "Conv1DCommon.cuh"

#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif
#define MAX_KL 33

extern "C" __global__ void cut_main(const cut_conv1d_t* __restrict__ input_data,
                                    const cut_conv1d_t* __restrict__ weight_data,
                                    cut_conv1d_t* __restrict__ output_data,
                                    PushConstants pc) {
    __shared__ cut_conv1d_t sharedInput[TILE_SIZE + MAX_KL - 1];

    uint L_out = (pc.L_in + 2 * pc.padding - pc.kL) / pc.stride + 1;
    uint inAlignedL = (pc.L_in + 3) & ~3u;
    uint outAlignedL = (L_out + 3) & ~3u;
    uint weightAlignedKL = (pc.kL + 3) & ~3u;

    // x tiles over L_out, y encodes (n, c_out)
    uint l_out = blockIdx.x * TILE_SIZE + threadIdx.x;
    uint linear_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint c_out = linear_y % pc.C_out;
    uint n = linear_y / pc.C_out;

    bool active = (l_out < L_out);

    cut_conv1d_t sum = (cut_conv1d_t)(0);

    // Process one input channel at a time using shared memory
    for (uint ci = 0; ci < pc.C_in; ci++) {
        uint sharedLen = min(TILE_SIZE + pc.kL - 1u, (uint)(TILE_SIZE + MAX_KL - 1));
        int baseL = int(blockIdx.x * TILE_SIZE * pc.stride) - int(pc.padding);

        // Cooperatively load input segment + halo into shared memory
        for (uint s = threadIdx.x; s < sharedLen; s += TILE_SIZE) {
            int il = baseL + int(s);
            cut_conv1d_t val = (cut_conv1d_t)(0);
            if (il >= 0 && il < int(pc.L_in)) {
                uint in_idx = n * pc.C_in * inAlignedL + ci * inAlignedL + uint(il);
                val = input_data[in_idx];
            }
            sharedInput[s] = val;
        }
        __syncthreads();

        // Compute convolution from shared memory
        if (active) {
            uint localL = threadIdx.x * pc.stride;
            for (uint kl = 0; kl < pc.kL; kl++) {
                if (localL + kl < sharedLen) {
                    uint w_idx = c_out * pc.C_in * weightAlignedKL + ci * weightAlignedKL + kl;
                    sum += sharedInput[localL + kl] * weight_data[w_idx];
                }
            }
        }

        __syncthreads();
    }

    if (active) {
        uint out_idx = n * pc.C_out * outAlignedL + c_out * outAlignedL + l_out;
        output_data[out_idx] = sum;
    }
}
