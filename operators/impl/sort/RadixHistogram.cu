// Native CUDA counterpart of RadixHistogram.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

#define WG_SIZE 256
#define RADIX 16

struct PushConstants {
    uint numElements;
    uint bitOffset;
    uint groupCount;
};

extern "C" __global__ void cut_main(const uint* __restrict__ keys,
                                    uint* __restrict__ histogram,
                                    PushConstants pc) {
    uint tid = threadIdx.x;
    uint gid = blockIdx.x;
    __shared__ uint localHist[RADIX];

    // Clear shared histogram
    if (tid < RADIX) {
        localHist[tid] = 0;
    }
    __syncthreads();

    // Count digits for this workgroup's elements
    for (uint i = gid * WG_SIZE + tid; i < pc.numElements; i += WG_SIZE * pc.groupCount) {
        uint digit = (keys[i] >> pc.bitOffset) & 0xFu;
        atomicAdd(&localHist[digit], 1u);
    }
    __syncthreads();

    // Write local histogram to global memory
    // Layout: histogram[digit * groupCount + gid]
    if (tid < RADIX) {
        histogram[tid * pc.groupCount + gid] = localHist[tid];
    }
}
