// Native CUDA counterpart of Dot.shader (per-workgroup partial dot product).
// Hash-aliased across dtypes: must stay pure float (the HLSL is float-typed);
// never branch on CUT_DTYPE_* here.
#include "ReduceCommon.cuh"
struct PushConstants { uint numElements; };
extern "C" __global__ void cut_main(const float* __restrict__ dataA, const float* __restrict__ dataB, float* __restrict__ dataOut, PushConstants pc) {
    uint tid = threadIdx.x;
    uint gid = blockIdx.x * 256u + tid;
    float v = (gid < pc.numElements) ? dataA[gid] * dataB[gid] : 0.0f;
    float s = cut_block_sum_f32(v, tid);
    if (tid == 0) {
        dataOut[blockIdx.x] = s;
    }
}
