#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint qDim;
    uint kvDim;
    uint headDim;
    uint halfDim;
    uint alignedKvDim;
};
[[vk::push_constant]] PushConstants pc;

// Input tensors
[[vk::binding(0, 0)]] StructuredBuffer<float> q;
[[vk::binding(1, 0)]] StructuredBuffer<float> k;
[[vk::binding(2, 0)]] StructuredBuffer<float> v;
// Runtime params [pos, seqLen]
[[vk::binding(3, 0)]] StructuredBuffer<uint> runtimeParams;
// Precomputed tables
[[vk::binding(4, 0)]] StructuredBuffer<float> cosTable;
[[vk::binding(5, 0)]] StructuredBuffer<float> sinTable;
// Cache buffers
#ifdef DTYPE_INPUT_IS_HALF
[[vk::binding(6, 0)]] RWStructuredBuffer<float16_t> kCache;
[[vk::binding(7, 0)]] RWStructuredBuffer<float16_t> vCache;
#else
[[vk::binding(6, 0)]] RWStructuredBuffer<float> kCache;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> vCache;
#endif
// Output tensor
[[vk::binding(8, 0)]] RWStructuredBuffer<float> qRoped;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint gid = DTid.x;
    uint totalWork = pc.qDim + 2 * pc.kvDim;
    if (gid >= totalWork) return;

    uint pos = runtimeParams[0];

    if (gid < pc.qDim) {
        // RoPE for q
        uint idxInHead = gid % pc.headDim;
        uint tableBase = pos * pc.halfDim;
        float x0, x1, cosVal, sinVal;

        if (idxInHead < pc.halfDim) {
            cosVal = cosTable[tableBase + idxInHead];
            sinVal = sinTable[tableBase + idxInHead];
            x0 = q[gid];
            x1 = q[gid + pc.halfDim];
            qRoped[gid] = x0 * cosVal - x1 * sinVal;
        } else {
            uint pairIdx = idxInHead - pc.halfDim;
            cosVal = cosTable[tableBase + pairIdx];
            sinVal = sinTable[tableBase + pairIdx];
            x0 = q[gid - pc.halfDim];
            x1 = q[gid];
            qRoped[gid] = x0 * sinVal + x1 * cosVal;
        }
    } else if (gid < pc.qDim + pc.kvDim) {
        // RoPE for k and write to kCache
        uint i = gid - pc.qDim;
        uint idxInHead = i % pc.headDim;
        uint tableBase = pos * pc.halfDim;
        float x0, x1, cosVal, sinVal;

        if (idxInHead < pc.halfDim) {
            cosVal = cosTable[tableBase + idxInHead];
            sinVal = sinTable[tableBase + idxInHead];
            x0 = k[i];
            x1 = k[i + pc.halfDim];
#ifdef DTYPE_INPUT_IS_HALF
            kCache[pos * pc.alignedKvDim + i] = float16_t(x0 * cosVal - x1 * sinVal);
#else
            kCache[pos * pc.alignedKvDim + i] = x0 * cosVal - x1 * sinVal;
#endif
        } else {
            uint pairIdx = idxInHead - pc.halfDim;
            cosVal = cosTable[tableBase + pairIdx];
            sinVal = sinTable[tableBase + pairIdx];
            x0 = k[i - pc.halfDim];
            x1 = k[i];
#ifdef DTYPE_INPUT_IS_HALF
            kCache[pos * pc.alignedKvDim + i] = float16_t(x0 * sinVal + x1 * cosVal);
#else
            kCache[pos * pc.alignedKvDim + i] = x0 * sinVal + x1 * cosVal;
#endif
        }
    } else {
        // Write v to vCache
        uint i = gid - pc.qDim - pc.kvDim;
#ifdef DTYPE_INPUT_IS_HALF
        vCache[pos * pc.alignedKvDim + i] = float16_t(v[i]);
#else
        vCache[pos * pc.alignedKvDim + i] = v[i];
#endif
    }
}
