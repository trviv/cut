#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256

// Batched K + V cache write for prefill, with K-side RoPE.
//
// Writes:
//   kCache[positions[i], :] = RoPE(kIn[i, kOffset : kOffset + kvDim])
//   vCache[positions[i], :] =        vIn[i, vOffset : vOffset + kvDim]
//
// for i in 0..N-1, where positions is a [N] uint buffer of absolute token
// positions in the sequence.
//
// This is the cache-write half of what BatchedFusedAttention used to do
// in a single dispatch — splitting it off lets us run cache write and
// attention as two separate dispatches, with the Vulkan barrier between
// them ensuring all cache writes complete before any attention head reads.
// (The original fused op had a cross-workgroup race where token j's
// attention could read token i's cache row before token i's write
// completed.)
//
// Dispatch: (1, N, 1) — one workgroup per token, all WG threads collaborate
// on writing the row. K-side does RoPE inline.
struct PushConstants {
    uint batchSize;
    uint kvDim;          // nKvHeads * headDim
    uint alignedKvDim;   // (kvDim + 3) & ~3
    uint headDim;
    uint halfDim;        // headDim / 2
    uint kStride;        // K input row stride (elements)
    uint vStride;        // V input row stride (elements)
    uint kOffset;        // Column offset of K within each input row
    uint vOffset;        // Column offset of V within each input row
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> kIn;
[[vk::binding(1, 0)]] StructuredBuffer<float> vIn;
#ifdef DTYPE_INPUT_IS_HALF
[[vk::binding(2, 0)]] RWStructuredBuffer<float16_t> kCache;
[[vk::binding(3, 0)]] RWStructuredBuffer<float16_t> vCache;
#else
[[vk::binding(2, 0)]] RWStructuredBuffer<float> kCache;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> vCache;
#endif
[[vk::binding(4, 0)]] StructuredBuffer<uint> positions;
[[vk::binding(5, 0)]] StructuredBuffer<float> cosTable;
[[vk::binding(6, 0)]] StructuredBuffer<float> sinTable;

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tokenIdx = Gid.y;
    uint tid = GTid.x;
    if (tokenIdx >= pc.batchSize) return;

    uint pos = positions[tokenIdx];
    uint tableBase = pos * pc.halfDim;
    uint kRowBase = tokenIdx * pc.kStride + pc.kOffset;
    uint vRowBase = tokenIdx * pc.vStride + pc.vOffset;
    uint cacheBase = pos * pc.alignedKvDim;

    // Write RoPE(K) to K cache
    for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
        uint idxInHead = d % pc.headDim;
        float kVal;
        if (idxInHead < pc.halfDim) {
            float cosVal = cosTable[tableBase + idxInHead];
            float sinVal = sinTable[tableBase + idxInHead];
            kVal = kIn[kRowBase + d] * cosVal
                 - kIn[kRowBase + d + pc.halfDim] * sinVal;
        } else {
            uint pairIdx = idxInHead - pc.halfDim;
            float cosVal = cosTable[tableBase + pairIdx];
            float sinVal = sinTable[tableBase + pairIdx];
            kVal = kIn[kRowBase + d - pc.halfDim] * sinVal
                 + kIn[kRowBase + d] * cosVal;
        }
#ifdef DTYPE_INPUT_IS_HALF
        kCache[cacheBase + d] = float16_t(kVal);
#else
        kCache[cacheBase + d] = kVal;
#endif
    }

    // Write V to V cache (no RoPE)
    for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
#ifdef DTYPE_INPUT_IS_HALF
        vCache[cacheBase + d] = float16_t(vIn[vRowBase + d]);
#else
        vCache[cacheBase + d] = vIn[vRowBase + d];
#endif
    }
}
