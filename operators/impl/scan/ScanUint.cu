// Native CUDA counterpart of ScanUint.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

struct PushConstants {
    uint numElements;
};

extern "C" __global__ void cut_main(uint* __restrict__ data, PushConstants pc) {
    uint sum = 0;
    for (uint i = 0; i < pc.numElements; i++) {
        uint val = data[i];
        data[i] = sum;
        sum += val;
    }
}
