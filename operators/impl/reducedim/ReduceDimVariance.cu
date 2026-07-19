// Native CUDA counterpart of ReduceDimVariance.shader — keep semantics in lockstep.
// Single-pass Welford variance; the HLSL groupshared merge tree is replaced by
// warp-shuffle Welford merges with cross-warp partials in shared memory.
#include "ComputeOpsShared.h"
#include "ReduceDimCommon.cuh"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

static __device__ __forceinline__ void cut_welford_merge(float &a_mean, float &a_m2, float &a_count,
                                                         float b_mean, float b_m2, float b_count) {
    float new_count = a_count + b_count;
    if (new_count > 0.0f) {
        float delta = b_mean - a_mean;
        a_mean = a_mean + delta * b_count / new_count;
        a_m2 = a_m2 + b_m2 + delta * delta * a_count * b_count / new_count;
    }
    a_count = new_count;
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

    // Phase 1: per-thread Welford accumulation (identical to the shader)
    float t_mean = 0.0f, t_m2 = 0.0f, t_count = 0.0f;
    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        float v = dataIn[outer * pc.inOuterStride + r * pc.inReduceStride + inner];
        t_count += 1.0f;
        float delta1 = v - t_mean;
        t_mean += delta1 / t_count;
        float delta2 = v - t_mean;
        t_m2 += delta1 * delta2;
    }

    // Phase 2: warp shuffle Welford merge, then cross-warp partials
    #pragma unroll
    for (uint off = 16u; off > 0u; off >>= 1) {
        float b_mean = __shfl_down_sync(0xffffffffu, t_mean, off);
        float b_m2 = __shfl_down_sync(0xffffffffu, t_m2, off);
        float b_count = __shfl_down_sync(0xffffffffu, t_count, off);
        cut_welford_merge(t_mean, t_m2, t_count, b_mean, b_m2, b_count);
    }

    __shared__ float sMean[WG_SIZE / 32];
    __shared__ float sM2[WG_SIZE / 32];
    __shared__ float sCount[WG_SIZE / 32];
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    if (lane == 0u) {
        sMean[warp] = t_mean;
        sM2[warp] = t_m2;
        sCount[warp] = t_count;
    }
    __syncthreads();

    // Phase 3: thread 0 folds the per-warp partials and writes variance = M2 / count
    if (tid == 0u) {
        float acc_mean = sMean[0];
        float acc_m2 = sM2[0];
        float acc_count = sCount[0];
        #pragma unroll
        for (uint w = 1u; w < WG_SIZE / 32; ++w) {
            cut_welford_merge(acc_mean, acc_m2, acc_count, sMean[w], sM2[w], sCount[w]);
        }
        dataOut[outIdx] = (acc_count > 0.0f) ? (acc_m2 / acc_count) : 0.0f;
    }
}
