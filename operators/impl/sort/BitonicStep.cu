// Native CUDA counterpart of BitonicStep.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

struct PushConstants {
    uint numElements;
    uint outerStep;
    uint innerStep;
};

extern "C" __global__ void cut_main(float* __restrict__ keys,
                                    uint* __restrict__ vals,
                                    PushConstants pc) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint ixj = idx ^ pc.innerStep;

    if (ixj <= idx || idx >= pc.numElements || ixj >= pc.numElements) {
        return;
    }

    bool ascending = ((idx & pc.outerStep) == 0);

    float keyI = keys[idx];
    float keyJ = keys[ixj];

    if ((ascending && keyI > keyJ) || (!ascending && keyI < keyJ)) {
        keys[idx] = keyJ;
        keys[ixj] = keyI;
        uint valI = vals[idx];
        uint valJ = vals[ixj];
        vals[idx] = valJ;
        vals[ixj] = valI;
    }
}
