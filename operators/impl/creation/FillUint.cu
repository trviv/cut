// Native CUDA counterpart of FillUint.shader. Fully dtype-agnostic.

#include "ComputeOpsShared.h"

struct PushConstants {
    uint numElements;
    uint fillValue;
};

extern "C" __global__ void cut_main(uint* __restrict__ dataOut, PushConstants pc) {
    uint3 DTid;
    DTid.x = blockIdx.x * blockDim.x + threadIdx.x; DTid.y = blockIdx.y * blockDim.y + threadIdx.y; DTid.z = blockIdx.z * blockDim.z + threadIdx.z;

    uint idx = DTid.x;
    if (idx < pc.numElements) {
        dataOut[idx] = pc.fillValue;
    }
}
