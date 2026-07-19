// Native CUDA counterpart of ReduceDimRMS.shader — keep semantics in lockstep.
// Single-pass RMS = sqrt(sum(x^2) / reduceSize); the HLSL groupshared tree is
// replaced by a warp-shuffle sum with cross-warp partials in shared memory.
#include "ComputeOpsShared.h"
#include "ReduceDimCommon.cuh"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

extern "C" __global__ void cut_main(const float* __restrict__ dataIn,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint outIdx = blockIdx.x;
    uint tid = threadIdx.x;
    uint numOutputs = pc.outerSize * pc.innerSize;
    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    // Phase 1: per-thread sum of squares (identical to the shader)
    float localSum = 0.0f;
    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        float v = dataIn[outer * pc.inOuterStride + r * pc.inReduceStride + inner];
        localSum += v * v;
    }

    // Phase 2: warp shuffle sum, then cross-warp partials
    #pragma unroll
    for (uint off = 16u; off > 0u; off >>= 1) {
        localSum += __shfl_down_sync(0xffffffffu, localSum, off);
    }

    __shared__ float warpPartials[WG_SIZE / 32];
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    if (lane == 0u) {
        warpPartials[warp] = localSum;
    }
    __syncthreads();

    // Phase 3: thread 0 folds the per-warp partials and writes RMS
    if (tid == 0u) {
        float total = warpPartials[0];
        #pragma unroll
        for (uint w = 1u; w < WG_SIZE / 32; ++w) {
            total += warpPartials[w];
        }
        dataOut[outIdx] = sqrtf(total / (float)pc.reduceSize);
    }
}
