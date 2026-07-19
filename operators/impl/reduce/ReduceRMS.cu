// Native CUDA counterpart of ReduceRMS.shader (single-workgroup single-pass
// RMS = sqrt(sum(x^2) / n), alignment-aware indexing).
#include "ReduceCommon.cuh"
struct PushConstants { uint numElements; uint actualInner; uint alignedInner; };
extern "C" __global__ void cut_main(const float* __restrict__ dataIn, float* __restrict__ dataOut, PushConstants pc) {
    uint tid = threadIdx.x;
    float localSum = 0.0f;
    for (uint i = tid; i < pc.numElements; i += 256u) {
        uint row = i / pc.actualInner;
        uint col = i % pc.actualInner;
        uint idx = row * pc.alignedInner + col;
        float val = dataIn[idx];
        localSum += val * val;
    }
    float s = cut_block_sum_f32(localSum, tid);
    if (tid == 0) {
        dataOut[0] = sqrtf(s / (float)pc.numElements);
    }
}
