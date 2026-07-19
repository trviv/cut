// Native CUDA counterpart of ScanPropagate.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

#define WG_SIZE 256

struct PushConstants {
    uint numElements;
};

extern "C" __global__ void cut_main(const float* __restrict__ partialSums,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint gid = blockIdx.x;
    uint idx = gid * WG_SIZE + threadIdx.x;
    if (idx < pc.numElements && gid > 0) {
        dataOut[idx] += partialSums[gid];
    }
}
