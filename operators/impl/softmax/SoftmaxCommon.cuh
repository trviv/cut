// Native CUDA counterpart shared by Softmax.cu / LogSoftmax.cu — push constants,
// dtype defaults, and the block-wide online (max, sumexp) reduction.
//
// The reduction keeps the exact shared-memory tree of the HLSL shaders
// (stride WG_SIZE/2 -> 1) so results stay numerically identical to Vulkan —
// do NOT rewrite it with warp shuffles.
#pragma once

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif
#ifndef CUT_DTYPE_INPUT_IS_HALF
#define CUT_DTYPE_INPUT_IS_HALF 0
#endif

struct PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

// Block-wide online-normalizer merge; all WG_SIZE threads must call it and the
// (max, sumexp) result is broadcast to every thread.
static __device__ __forceinline__ void cut_block_online_reduce(float local_m, float local_d,
                                                               float &global_m, float &global_d) {
    __shared__ float shared_max[WG_SIZE];
    __shared__ float shared_sumexp[WG_SIZE];
    uint tid = threadIdx.x;
    shared_max[tid] = local_m;
    shared_sumexp[tid] = local_d;
    __syncthreads();
    for (uint stride = WG_SIZE / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float a_m = shared_max[tid];
            float a_d = shared_sumexp[tid];
            float b_m = shared_max[tid + stride];
            float b_d = shared_sumexp[tid + stride];
            float new_m = fmaxf(a_m, b_m);
            shared_sumexp[tid] = a_d * expf(a_m - new_m) + b_d * expf(b_m - new_m);
            shared_max[tid] = new_m;
        }
        __syncthreads();
    }
    global_m = shared_max[0];
    global_d = shared_sumexp[0];
}
