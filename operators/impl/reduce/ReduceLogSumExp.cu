// Native CUDA counterpart of ReduceLogSumExp.shader (single-workgroup
// online-normalizer logsumexp, Milakov & Gimelshein 2018). The HLSL
// groupshared tree is replaced by a warp-shuffle merge; the merge is
// commutative and (-FLT_MAX, 0) is its identity, so the xor butterfly is safe.
#include "ReduceCommon.cuh"
struct PushConstants { uint numElements; uint actualInner; uint alignedInner; };

__device__ __forceinline__ void cut_lse_merge(float &a_m, float &a_d, float b_m, float b_d) {
    float new_m = fmaxf(a_m, b_m);
    a_d = a_d * expf(a_m - new_m) + b_d * expf(b_m - new_m);
    a_m = new_m;
}

extern "C" __global__ void cut_main(const float* __restrict__ dataIn, float* __restrict__ dataOut, PushConstants pc) {
    uint tid = threadIdx.x;

    // Phase 1: per-thread online (max, sumexp) accumulation via strided loop
    float local_m = -3.402823466e+38f;  // -FLT_MAX
    float local_d = 0.0f;
    for (uint i = tid; i < pc.numElements; i += 256u) {
        uint row = i / pc.actualInner;
        uint col = i % pc.actualInner;
        uint idx = row * pc.alignedInner + col;
        float x = dataIn[idx];
        float new_m = fmaxf(local_m, x);
        local_d = local_d * expf(local_m - new_m) + expf(x - new_m);
        local_m = new_m;
    }

    // Phase 2: warp-shuffle merge, then cross-warp merge via shared staging
    __shared__ float s_m[CUT_REDUCE_WARPS];
    __shared__ float s_d[CUT_REDUCE_WARPS];
    float m = local_m;
    float d = local_d;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float p_m = __shfl_xor_sync(0xffffffffu, m, offset);
        float p_d = __shfl_xor_sync(0xffffffffu, d, offset);
        cut_lse_merge(m, d, p_m, p_d);
    }
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    if (lane == 0) {
        s_m[warp] = m;
        s_d[warp] = d;
    }
    __syncthreads();
    if (warp == 0) {
        m = (lane < CUT_REDUCE_WARPS) ? s_m[lane] : -3.402823466e+38f;
        d = (lane < CUT_REDUCE_WARPS) ? s_d[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float p_m = __shfl_xor_sync(0xffffffffu, m, offset);
            float p_d = __shfl_xor_sync(0xffffffffu, d, offset);
            cut_lse_merge(m, d, p_m, p_d);
        }
    }

    // Phase 3: logsumexp = max + log(sumexp)
    if (tid == 0) {
        dataOut[0] = m + logf(d);
    }
}
