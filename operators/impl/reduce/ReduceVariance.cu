// Native CUDA counterpart of ReduceVariance.shader (single-workgroup
// single-pass Welford population variance). The HLSL groupshared tree is
// replaced by a warp-shuffle Welford merge; the merge is commutative and
// (0,0,0) is its identity, so the xor butterfly is safe.
#include "ReduceCommon.cuh"
struct PushConstants { uint numElements; uint actualInner; uint alignedInner; };

__device__ __forceinline__ void cut_welford_merge(float &a_mean, float &a_m2, float &a_count, float b_mean, float b_m2, float b_count) {
    float nc = a_count + b_count;
    if (nc > 0.0f) {
        float delta = b_mean - a_mean;
        a_mean = a_mean + delta * b_count / nc;
        a_m2 = a_m2 + b_m2 + delta * delta * a_count * b_count / nc;
    }
    a_count = nc;
}

extern "C" __global__ void cut_main(const float* __restrict__ dataIn, float* __restrict__ dataOut, PushConstants pc) {
    uint tid = threadIdx.x;

    // Phase 1: per-thread Welford accumulation via strided loop
    float t_mean = 0.0f;
    float t_m2 = 0.0f;
    float t_count = 0.0f;
    for (uint i = tid; i < pc.numElements; i += 256u) {
        uint row = i / pc.actualInner;
        uint col = i % pc.actualInner;
        uint idx = row * pc.alignedInner + col;
        float val = dataIn[idx];
        t_count += 1.0f;
        float delta1 = val - t_mean;
        t_mean += delta1 / t_count;
        float delta2 = val - t_mean;
        t_m2 += delta1 * delta2;
    }

    // Phase 2: warp-shuffle merge, then cross-warp merge via shared staging
    __shared__ float s_mean[CUT_REDUCE_WARPS];
    __shared__ float s_m2[CUT_REDUCE_WARPS];
    __shared__ float s_count[CUT_REDUCE_WARPS];
    float m = t_mean;
    float m2 = t_m2;
    float count = t_count;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float p_mean = __shfl_xor_sync(0xffffffffu, m, offset);
        float p_m2 = __shfl_xor_sync(0xffffffffu, m2, offset);
        float p_count = __shfl_xor_sync(0xffffffffu, count, offset);
        cut_welford_merge(m, m2, count, p_mean, p_m2, p_count);
    }
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    if (lane == 0) {
        s_mean[warp] = m;
        s_m2[warp] = m2;
        s_count[warp] = count;
    }
    __syncthreads();
    if (warp == 0) {
        m = (lane < CUT_REDUCE_WARPS) ? s_mean[lane] : 0.0f;
        m2 = (lane < CUT_REDUCE_WARPS) ? s_m2[lane] : 0.0f;
        count = (lane < CUT_REDUCE_WARPS) ? s_count[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float p_mean = __shfl_xor_sync(0xffffffffu, m, offset);
            float p_m2 = __shfl_xor_sync(0xffffffffu, m2, offset);
            float p_count = __shfl_xor_sync(0xffffffffu, count, offset);
            cut_welford_merge(m, m2, count, p_mean, p_m2, p_count);
        }
    }

    // Phase 3: variance = M2 / count
    if (tid == 0) {
        dataOut[0] = (count > 0.0f) ? m2 / count : 0.0f;
    }
}
