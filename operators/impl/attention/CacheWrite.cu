// Native CUDA counterpart of CacheWrite.shader — keep semantics in lockstep.
//
// NOTE: the Int8 and Float32 variants of this shader hash-alias (identical
// SPIR-V) and share one compiled kernel (built with the Float32 defines), so
// this code must not branch on any Int8 macro. Branching on
// CUT_DTYPE_INPUT_IS_HALF (via cut_kv_t) is fine: the Float16 variant has its
// own hash.
#include "ComputeOpsShared.h"
#include "AttentionCommon.cuh"

struct PushConstants {
    uint kvDim;        // Elements per row (actual)
    uint alignedKvDim; // Aligned stride for 2D cache
};

extern "C" __global__ void cut_main(const float* __restrict__ newData,
                                    const uint* __restrict__ runtimeParams,
                                    cut_kv_t* __restrict__ cache,
                                    PushConstants pc) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pc.kvDim) return;

    uint pos = runtimeParams[0];
    cache[pos * pc.alignedKvDim + gid] = cut_kv_store(newData[gid]);
}
