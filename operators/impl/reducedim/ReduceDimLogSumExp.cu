// Native CUDA counterpart of ReduceDimLogSumExp.shader — keep semantics in lockstep.
// Single-pass LogSumExp with an online (max, sumexp) normalizer; the HLSL
// groupshared merge tree is replaced by warp-shuffle online merges with
// cross-warp partials in shared memory.
#include "ComputeOpsShared.h"
#include "ReduceDimCommon.cuh"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

static __device__ __forceinline__ void cut_online_merge(float &m, float &d, float b_m, float b_d) {
    float new_m = fmaxf(m, b_m);
    d = d * expf(m - new_m) + b_d * expf(b_m - new_m);
    m = new_m;
}

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

    // Phase 1: per-thread online (max, sumexp) accumulation (identical to the shader)
    float local_m = -3.402823466e+38f;  // -FLT_MAX
    float local_d = 0.0f;
    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        float x = dataIn[outer * pc.inOuterStride + r * pc.inReduceStride + inner];
        float new_m = fmaxf(local_m, x);
        local_d = local_d * expf(local_m - new_m) + expf(x - new_m);
        local_m = new_m;
    }

    // Phase 2: warp shuffle online merge, then cross-warp partials
    #pragma unroll
    for (uint off = 16u; off > 0u; off >>= 1) {
        float b_m = __shfl_down_sync(0xffffffffu, local_m, off);
        float b_d = __shfl_down_sync(0xffffffffu, local_d, off);
        cut_online_merge(local_m, local_d, b_m, b_d);
    }

    __shared__ float sMax[WG_SIZE / 32];
    __shared__ float sSum[WG_SIZE / 32];
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    if (lane == 0u) {
        sMax[warp] = local_m;
        sSum[warp] = local_d;
    }
    __syncthreads();

    // Phase 3: thread 0 folds the per-warp partials and writes logsumexp = max + log(sumexp)
    if (tid == 0u) {
        float m = sMax[0];
        float d = sSum[0];
        #pragma unroll
        for (uint w = 1u; w < WG_SIZE / 32; ++w) {
            cut_online_merge(m, d, sMax[w], sSum[w]);
        }
        dataOut[outIdx] = m + logf(d);
    }
}
