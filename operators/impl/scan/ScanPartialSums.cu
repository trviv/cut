// Native CUDA counterpart of ScanPartialSums.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

struct PushConstants {
    uint numGroups;
};

extern "C" __global__ void cut_main(float* __restrict__ partialSums, PushConstants pc) {
    float sum = 0.0f;
    for (uint i = 0; i < pc.numGroups; i++) {
        float val = partialSums[i];
        partialSums[i] = sum;
        sum += val;
    }
}
