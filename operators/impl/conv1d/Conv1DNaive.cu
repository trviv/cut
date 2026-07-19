// Native CUDA counterpart of Conv1DNaive.shader (block dims 256x1x1) — keep
// semantics in lockstep.
#include "Conv1DCommon.cuh"

extern "C" __global__ void cut_main(const cut_conv1d_t* __restrict__ input_data,
                                    const cut_conv1d_t* __restrict__ weight_data,
                                    cut_conv1d_t* __restrict__ output_data,
                                    PushConstants pc) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint L_out = (pc.L_in + 2 * pc.padding - pc.kL) / pc.stride + 1;
    uint inAlignedL = (pc.L_in + 3) & ~3u;
    uint outAlignedL = (L_out + 3) & ~3u;
    uint weightAlignedKL = (pc.kL + 3) & ~3u;

    // Decode linear index into (n, c_out, l_out)
    uint l_out = gid % L_out;
    uint nc = gid / L_out;
    uint c_out = nc % pc.C_out;
    uint n = nc / pc.C_out;

    if (n >= pc.batchSize) return;

    cut_conv1d_t sum = (cut_conv1d_t)(0);

    for (uint ci = 0; ci < pc.C_in; ci++) {
        for (uint kl = 0; kl < pc.kL; kl++) {
            int l_in = int(l_out * pc.stride + kl) - int(pc.padding);
            if (l_in < 0 || l_in >= int(pc.L_in)) continue;

            uint in_idx = n * pc.C_in * inAlignedL
                        + ci * inAlignedL
                        + uint(l_in);

            uint w_idx = c_out * pc.C_in * weightAlignedKL
                       + ci * weightAlignedKL
                       + kl;

            sum += input_data[in_idx] * weight_data[w_idx];
        }
    }

    uint out_idx = n * pc.C_out * outAlignedL
                 + c_out * outAlignedL
                 + l_out;

    output_data[out_idx] = sum;
}
