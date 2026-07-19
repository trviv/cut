// Native CUDA counterpart shared by RMSNorm.cu / ExtendedRMSNorm.cu — push
// constants, dtype defaults, and the block-wide float sum reduction.
#pragma once

#ifndef CUT_SCALAR_DTYPE_INPUT
#define CUT_SCALAR_DTYPE_INPUT float
#endif
#ifndef CUT_DTYPE_INPUT_IS_HALF
#define CUT_DTYPE_INPUT_IS_HALF 0
#endif

struct PushConstants {
    uint dim;         // Actual vector dimension (innermost)
    uint alignedDim;  // Padded to multiple of 4 (row stride for 2D)
    float eps;        // Normalization epsilon (1e-5)
};

// Block-wide sum over the 256-thread block (8 warps), result broadcast to every
// thread. Replaces the HLSL groupshared tree reduction with warp shuffles.
static __device__ __forceinline__ float cut_block_sum_broadcast(float v) {
    __shared__ float warpPartials[8];
    __shared__ float totalShared;
    uint tid = threadIdx.x;
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    #pragma unroll
    for (uint off = 16u; off > 0u; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0u) warpPartials[warp] = v;
    __syncthreads();
    if (tid == 0u) {
        float total = warpPartials[0];
        #pragma unroll
        for (uint w = 1u; w < 8u; ++w) total += warpPartials[w];
        totalShared = total;
    }
    __syncthreads();
    return totalShared;
}
