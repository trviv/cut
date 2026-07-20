// Native CUDA counterpart of MaxPool2DNaive.shader (block dims 16x16x1;
// 4-wide vectorized output) — keep semantics in lockstep.
#include "MaxPool2DCommon.cuh"

extern "C" __global__ void cut_main(const cut_pool_t* __restrict__ input_data,
                                    cut_pool_vec* __restrict__ output_data,
                                    PushConstants pc) {
    uint w_out4 = blockIdx.x * blockDim.x + threadIdx.x;
    uint linear_y = blockIdx.y * blockDim.y + threadIdx.y;

    uint H_out = (pc.H_in + 2 * pc.padH - pc.kernelH) / pc.strideH + 1;
    uint W_out = (pc.W_in + 2 * pc.padW - pc.kernelW) / pc.strideW + 1;
    uint inAlignedW = (pc.W_in + 3) & ~3u;
    uint outAlignedW4 = ((W_out + 3) & ~3u) / 4;

    // Decode linear_y into (n, c, h_out)
    uint h_out = linear_y % H_out;
    uint nc = linear_y / H_out;
    uint c = nc % pc.C;
    uint n = nc / pc.C;

    if (w_out4 >= outAlignedW4 || n >= pc.N) return;

    uint baseW = w_out4 * 4;
    cut_pool_vec maxVals = CUT_POOL_VCAST((cut_pool_t)(-1.0e38));

    for (uint kh = 0; kh < pc.kernelH; kh++) {
        int h_in = int(h_out * pc.strideH + kh) - int(pc.padH);
        if (h_in < 0 || h_in >= int(pc.H_in)) continue;

        for (uint kw = 0; kw < pc.kernelW; kw++) {
            for (uint i = 0; i < 4; i++) {
                uint w_out = baseW + i;
                if (w_out >= W_out) continue;
                int w_in = int(w_out * pc.strideW + kw) - int(pc.padW);
                if (w_in < 0 || w_in >= int(pc.W_in)) continue;

                uint in_idx = n * pc.C * pc.H_in * inAlignedW
                            + c * pc.H_in * inAlignedW
                            + uint(h_in) * inAlignedW
                            + uint(w_in);

                maxVals[i] = max(maxVals[i], input_data[in_idx]);
            }
        }
    }

    // Zero out padding positions
    for (uint i = 0; i < 4; i++) {
        if (baseW + i >= W_out) maxVals[i] = (cut_pool_t)(0);
    }

    uint out_idx = n * pc.C * H_out * outAlignedW4
                 + c * H_out * outAlignedW4
                 + h_out * outAlignedW4
                 + w_out4;

    output_data[out_idx] = maxVals;
}
