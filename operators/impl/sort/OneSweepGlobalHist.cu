// Native CUDA counterpart of OneSweepGlobalHist.shader (a stub kept only for
// SPIR-V hashing). OneSweep global histogram: one pass over the original keys
// counting all NUM_PASSES 8-bit digit-places at once into
// globalHist[pass * RADIX + digit]. Order-independent, so computed once.
#include "ComputeOpsShared.h"

#define WG_SIZE 256
#define RADIX 256
#define NUM_PASSES 4

struct PushConstants {
    uint numElements;
};

extern "C" __global__ void cut_main(const uint* __restrict__ keys,
                                    uint* __restrict__ globalHist,
                                    PushConstants pc) {
    uint tid = threadIdx.x;
    __shared__ uint localHist[NUM_PASSES * RADIX];

    for (uint i = tid; i < NUM_PASSES * RADIX; i += WG_SIZE) {
        localHist[i] = 0u;
    }
    __syncthreads();

    uint stride = WG_SIZE * gridDim.x;
    for (uint i = blockIdx.x * WG_SIZE + tid; i < pc.numElements; i += stride) {
        uint key = keys[i];
#pragma unroll
        for (uint p = 0; p < NUM_PASSES; p++) {
            uint digit = (key >> (p * 8u)) & 0xFFu;
            atomicAdd(&localHist[p * RADIX + digit], 1u);
        }
    }
    __syncthreads();

    // Merge this block's partial histogram into the global one.
    for (uint i = tid; i < NUM_PASSES * RADIX; i += WG_SIZE) {
        uint v = localHist[i];
        if (v != 0u) {
            atomicAdd(&globalHist[i], v);
        }
    }
}
