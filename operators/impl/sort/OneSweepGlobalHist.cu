// Native CUDA counterpart of OneSweepGlobalHist.shader. OneSweep global
// histogram: one pass over the original keys counting all NUM_PASSES 8-bit
// digit-places at once into globalHist[pass * RADIX + digit]. Order-independent,
// so computed once.
//
// PERSISTENT GRID. Each block owns a private NUM_PASSES * RADIX shared histogram
// that it must zero on entry and merge into global on exit — 1024 shared stores
// plus up to 1024 global atomics of fixed cost per block. With one block per 256
// elements that overhead is 4 shared stores and 4 global atomics PER ELEMENT and
// it swamps the actual counting: the kernel measured 92 GB/s, an eighth of what
// a stream of this shape should reach. The grid is capped by the caller instead
// (see kHistMaxBlocks in SortOp.cpp) and each block grid-strides over its share,
// which spreads the same fixed cost over thousands of elements.
//
// The stride loop is unrolled by 4 with independent loads so the four requests
// are in flight together; the counting itself is latency-tolerant shared-memory
// atomics, so without that the loop runs one dependent global load at a time.
#include "ComputeOpsShared.h"

#define WG_SIZE 256
#define RADIX 256
#define NUM_PASSES 4

struct PushConstants {
    uint numElements;
    uint numGroups;
};

__device__ __forceinline__ void countKey(uint* localHist, uint key) {
#pragma unroll
    for (uint p = 0; p < NUM_PASSES; p++) {
        const uint digit = (key >> (p * 8u)) & 0xFFu;
        atomicAdd(&localHist[p * RADIX + digit], 1u);
    }
}

extern "C" __global__ void cut_main(const uint* __restrict__ keys,
                                    uint* __restrict__ globalHist,
                                    PushConstants pc) {
    uint tid = threadIdx.x;
    __shared__ uint localHist[NUM_PASSES * RADIX];

    for (uint i = tid; i < NUM_PASSES * RADIX; i += WG_SIZE) {
        localHist[i] = 0u;
    }
    __syncthreads();

    const uint stride = WG_SIZE * pc.numGroups;
    const uint n = pc.numElements;
    uint i = blockIdx.x * WG_SIZE + tid;
    for (; i + 3u * stride < n; i += 4u * stride) {
        const uint k0 = keys[i];
        const uint k1 = keys[i + stride];
        const uint k2 = keys[i + 2u * stride];
        const uint k3 = keys[i + 3u * stride];
        countKey(localHist, k0);
        countKey(localHist, k1);
        countKey(localHist, k2);
        countKey(localHist, k3);
    }
    for (; i < n; i += stride) {
        countKey(localHist, keys[i]);
    }
    __syncthreads();

    // Merge this block's partial histogram into the global one.
    for (uint j = tid; j < NUM_PASSES * RADIX; j += WG_SIZE) {
        uint v = localHist[j];
        if (v != 0u) {
            atomicAdd(&globalHist[j], v);
        }
    }
}
