// Native CUDA counterpart of BitonicPadInit.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

struct PushConstants {
    uint numElements;
    uint paddedSize;
};

extern "C" __global__ void cut_main(const float* __restrict__ srcKeys,
                                    const uint* __restrict__ srcVals,
                                    float* __restrict__ dstKeys,
                                    uint* __restrict__ dstVals,
                                    PushConstants pc) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.paddedSize) return;
    if (idx < pc.numElements) {
        dstKeys[idx] = srcKeys[idx];
        dstVals[idx] = srcVals[idx];
    } else {
        dstKeys[idx] = 3.402823466e+38f;
        dstVals[idx] = 0xFFFFFFFFu;
    }
}
