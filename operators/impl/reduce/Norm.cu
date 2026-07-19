// Native CUDA counterpart of Norm.shader (single-workgroup L2 norm).
// Hash-aliased across dtypes: must stay pure float (the HLSL is float-typed);
// never branch on CUT_DTYPE_* here.
#include "ReduceCommon.cuh"
struct PushConstants { uint numElements; };
extern "C" __global__ void cut_main(const float* __restrict__ dataIn, float* __restrict__ dataOut, PushConstants pc) {
    uint tid = threadIdx.x;
    float localVal = 0.0f;
    for (uint i = tid; i < pc.numElements; i += 256u) {
        float val = dataIn[i];
        localVal += val * val;
    }
    float s = cut_block_sum_f32(localVal, tid);
    if (tid == 0) {
        dataOut[0] = sqrtf(s);
    }
}
