// Native CUDA counterpart of BitonicCopyBack.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

struct PushConstants {
    uint numElements;
};

extern "C" __global__ void cut_main(const float* __restrict__ srcKeys,
                                    const uint* __restrict__ srcVals,
                                    float* __restrict__ dstKeys,
                                    uint* __restrict__ dstVals,
                                    PushConstants pc) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.numElements) return;
    dstKeys[idx] = srcKeys[idx];
    dstVals[idx] = srcVals[idx];
}
