#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256
#define MAX_SEQ_LEN 2048

// Batched attention compute (Phases 1-4 of the old BatchedFusedAttention).
// Reads K/V cache that must have already been populated by a prior dispatch
// (see BatchedKVCacheWrite). Applies RoPE to Q inline.
//
// Output: dataOut[N, nHeads * headDim]
// Dispatch: (nHeads, batchSize, 1)
//
// One workgroup per (head, token). Within a workgroup, threads collaborate
// on the per-token softmax over `seqLen = pos+1` cache positions.
struct PushConstants {
    uint batchSize;
    uint nHeads;
    uint nKvHeads;
    uint headDim;
    uint alignedKvDim;
    uint nRep;           // nHeads / nKvHeads (GQA)
    float scale;
    uint halfDim;
    uint qStride;
    uint qOffset;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> qIn;
#ifdef DTYPE_INPUT_IS_HALF
[[vk::binding(1, 0)]] StructuredBuffer<float16_t> kCache;
[[vk::binding(2, 0)]] StructuredBuffer<float16_t> vCache;
#else
[[vk::binding(1, 0)]] StructuredBuffer<float> kCache;
[[vk::binding(2, 0)]] StructuredBuffer<float> vCache;
#endif
[[vk::binding(3, 0)]] StructuredBuffer<uint> positions;
[[vk::binding(4, 0)]] StructuredBuffer<float> cosTable;
[[vk::binding(5, 0)]] StructuredBuffer<float> sinTable;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> dataOut;

groupshared float sharedQ[128];
groupshared float sharedScores[MAX_SEQ_LEN];
groupshared float sharedReduce[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint h = Gid.x;
    uint tokenIdx = Gid.y;
    uint tid = GTid.x;
    uint kvH = h / pc.nRep;

    if (tokenIdx >= pc.batchSize) return;

    uint pos = positions[tokenIdx];
    uint seqLen = pos + 1;
    uint tableBase = pos * pc.halfDim;

    // Phase 1: RoPE(Q) for this head into shared memory
    uint qRowBase = tokenIdx * pc.qStride + pc.qOffset;
    if (tid < pc.headDim) {
        uint gid = h * pc.headDim + tid;
        uint idxInHead = tid;
        if (idxInHead < pc.halfDim) {
            float cosVal = cosTable[tableBase + idxInHead];
            float sinVal = sinTable[tableBase + idxInHead];
            sharedQ[tid] = qIn[qRowBase + gid] * cosVal
                         - qIn[qRowBase + gid + pc.halfDim] * sinVal;
        } else {
            uint pairIdx = idxInHead - pc.halfDim;
            float cosVal = cosTable[tableBase + pairIdx];
            float sinVal = sinTable[tableBase + pairIdx];
            sharedQ[tid] = qIn[qRowBase + gid - pc.halfDim] * sinVal
                         + qIn[qRowBase + gid] * cosVal;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: scores = dot(Q_h, K_t) * scale  for t in 0..seqLen-1
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        float d = 0.0;
        uint kBase = t * pc.alignedKvDim + kvH * pc.headDim;
        for (uint i = 0; i < pc.headDim; i++) {
            d += sharedQ[i] * float(kCache[kBase + i]);
        }
        sharedScores[t] = d * pc.scale;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: softmax
    float localMax = -1e30;
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        localMax = max(localMax, sharedScores[t]);
    }
    sharedReduce[tid] = localMax;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedReduce[tid] = max(sharedReduce[tid], sharedReduce[tid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float maxScore = sharedReduce[0];

    float localSum = 0.0;
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        sharedScores[t] = exp(sharedScores[t] - maxScore);
        localSum += sharedScores[t];
    }
    sharedReduce[tid] = localSum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedReduce[tid] += sharedReduce[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float sumExp = sharedReduce[0];

    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        sharedScores[t] /= sumExp;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 4: weighted sum of V
    uint outStride = pc.nHeads * pc.headDim;
    if (tid < pc.headDim) {
        float acc = 0.0;
        for (uint t = 0; t < seqLen; t++) {
            acc += sharedScores[t]
                 * float(vCache[t * pc.alignedKvDim + kvH * pc.headDim + tid]);
        }
        dataOut[tokenIdx * outStride + h * pc.headDim + tid] = acc;
    }
}
