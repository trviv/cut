#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint kvDim;        // Elements per row (actual)
    uint alignedKvDim; // Aligned stride for 2D cache
};
[[vk::push_constant]] PushConstants pc;

// New vector to write [kvDim] — always float (from matmul output)
[[vk::binding(0, 0)]] StructuredBuffer<float> newData;
// Runtime params [pos, seqLen]
[[vk::binding(1, 0)]] StructuredBuffer<uint> runtimeParams;
// Cache buffer [maxSeqLen, kvDim] — float or half depending on dtype
#ifdef DTYPE_INPUT_IS_HALF
[[vk::binding(2, 0)]] RWStructuredBuffer<float16_t> cache;
#else
[[vk::binding(2, 0)]] RWStructuredBuffer<float> cache;
#endif

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint gid = DTid.x;
    if (gid >= pc.kvDim) return;

    uint pos = runtimeParams[0];
#ifdef DTYPE_INPUT_IS_HALF
    cache[pos * pc.alignedKvDim + gid] = float16_t(newData[gid]);
#else
    cache[pos * pc.alignedKvDim + gid] = newData[gid];
#endif
}
