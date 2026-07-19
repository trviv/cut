// Native CUDA counterpart of ScanPerWg.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

#define WG_SIZE 256
#define NUM_WARPS 8

struct PushConstants {
    uint numElements;
    uint isExclusive;
};

extern "C" __global__ void cut_main(const float* __restrict__ dataIn,
                                    float* __restrict__ dataOut,
                                    float* __restrict__ partialSums,
                                    PushConstants pc) {
    uint tid = threadIdx.x;
    uint gid = blockIdx.x;
    uint idx = gid * WG_SIZE + tid;
    uint lane = tid & 31u;
    uint warp = tid >> 5;

    float val = (idx < pc.numElements) ? dataIn[idx] : 0.0f;

    // Inclusive scan within each warp via __shfl_up_sync
    float inclusive = val;
#pragma unroll
    for (uint offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xFFFFFFFFu, inclusive, offset);
        if (lane >= offset) inclusive += n;
    }

    __shared__ float warpSums[NUM_WARPS];
    if (lane == 31u) warpSums[warp] = inclusive;
    __syncthreads();

    // Sum of totals of preceding warps (read all slots unconditionally to stay uniform)
    float warpPrefix = 0.0f;
#pragma unroll
    for (uint w = 0; w < NUM_WARPS; ++w) {
        float s = warpSums[w];
        if (w < warp) warpPrefix += s;
    }
    inclusive += warpPrefix;

    // Exclusive value = previous element's inclusive value. The shuffle must run
    // unconditionally (outside any divergent branch), full mask.
    float exclusive = __shfl_up_sync(0xFFFFFFFFu, inclusive, 1);
    if (lane == 0u) exclusive = warpPrefix;  // 0.0f for tid == 0 since warpPrefix is 0 in warp 0

    if (idx < pc.numElements) {
        dataOut[idx] = (pc.isExclusive != 0u) ? exclusive : inclusive;
    }
    if (tid == WG_SIZE - 1) {
        partialSums[gid] = inclusive;
    }
}
