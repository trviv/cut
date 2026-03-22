#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256
#define MAX_SEQ_LEN 2048

struct PushConstants {
    uint batchSize;    // N (number of tokens in batch)
    uint nHeads;
    uint nKvHeads;
    uint headDim;
    uint kvDim;        // nKvHeads * headDim
    uint alignedKvDim; // (kvDim + 3) & ~3
    uint nRep;         // nHeads / nKvHeads (GQA)
    float scale;       // 1/sqrt(headDim)
    uint halfDim;      // headDim / 2 for RoPE
    uint qStride;      // Stride between rows in Q buffer (elements)
    uint kStride;      // Stride between rows in K buffer (elements)
    uint vStride;      // Stride between rows in V buffer (elements)
    uint qOffset;      // Offset to Q data within row (0 for separate, 0 for fused)
    uint kOffset;      // Offset to K data within row (0 for separate, qdim for fused)
    uint vOffset;      // Offset to V data within row (0 for separate, qdim+kvdim for fused)
};
[[vk::push_constant]] PushConstants pc;

// Q input [N, qStride] — always float (pre-RoPE)
[[vk::binding(0, 0)]] StructuredBuffer<float> qIn;
// K input [N, kStride] — always float (pre-RoPE)
[[vk::binding(1, 0)]] StructuredBuffer<float> kIn;
// V input [N, vStride] — always float
[[vk::binding(2, 0)]] StructuredBuffer<float> vIn;
// K cache [maxSeqLen, alignedKvDim]
#ifdef DTYPE_INPUT_IS_HALF
[[vk::binding(3, 0)]] RWStructuredBuffer<float16_t> kCache;
[[vk::binding(4, 0)]] RWStructuredBuffer<float16_t> vCache;
#else
[[vk::binding(3, 0)]] RWStructuredBuffer<float> kCache;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> vCache;
#endif
// Position per token [N]
[[vk::binding(5, 0)]] StructuredBuffer<uint> posBuffer;
// Precomputed RoPE tables
[[vk::binding(6, 0)]] StructuredBuffer<float> cosTable;
[[vk::binding(7, 0)]] StructuredBuffer<float> sinTable;
// Output [N, nHeads * headDim] — always float
[[vk::binding(8, 0)]] RWStructuredBuffer<float> dataOut;

groupshared float sharedQ[128];            // Max head_dim
groupshared float sharedScores[MAX_SEQ_LEN];
groupshared float sharedReduce[WG_SIZE];

// Dispatch: (nHeads * 256, batchSize, 1)
// Gid.x = head index, Gid.y = token index in batch
[numthreads(WG_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID,
          uint3 GTid : SV_GroupThreadID,
          uint3 Gid  : SV_GroupID) {
    uint h = Gid.x;          // Attention head index
    uint tokenIdx = Gid.y;   // Which token in the batch
    uint tid = GTid.x;       // Thread within workgroup
    uint kvH = h / pc.nRep;  // KV head index (GQA)

    if (tokenIdx >= pc.batchSize) return;

    uint pos = posBuffer[tokenIdx];
    uint seqLen = pos + 1;  // Causal: attend to positions [0..pos]
    uint tableBase = pos * pc.halfDim;

    // Phase 0: First head writes RoPE(K) and V to cache at this token's position
    if (h == 0) {
        uint kRowBase = tokenIdx * pc.kStride + pc.kOffset;
        uint vRowBase = tokenIdx * pc.vStride + pc.vOffset;

        // Write RoPE(K) to K cache
        for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
            uint idxInHead = d % pc.headDim;
            float kVal;
            if (idxInHead < pc.halfDim) {
                float cosVal = cosTable[tableBase + idxInHead];
                float sinVal = sinTable[tableBase + idxInHead];
                kVal = kIn[kRowBase + d] * cosVal - kIn[kRowBase + d + pc.halfDim] * sinVal;
            } else {
                uint pairIdx = idxInHead - pc.halfDim;
                float cosVal = cosTable[tableBase + pairIdx];
                float sinVal = sinTable[tableBase + pairIdx];
                kVal = kIn[kRowBase + d - pc.halfDim] * sinVal + kIn[kRowBase + d] * cosVal;
            }
#ifdef DTYPE_INPUT_IS_HALF
            kCache[pos * pc.alignedKvDim + d] = float16_t(kVal);
#else
            kCache[pos * pc.alignedKvDim + d] = kVal;
#endif
        }

        // Write V to V cache (no RoPE)
        for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
#ifdef DTYPE_INPUT_IS_HALF
            vCache[pos * pc.alignedKvDim + d] = float16_t(vIn[vRowBase + d]);
#else
            vCache[pos * pc.alignedKvDim + d] = vIn[vRowBase + d];
#endif
        }
    }

    // All heads must wait for cache write by head 0 of THIS token.
    // Since different tokens write to different cache positions, no cross-token
    // conflict. But heads within a token share the cache row.
    GroupMemoryBarrierWithGroupSync();

    // Phase 1: Load RoPE(Q) for this head into shared memory
    uint qRowBase = tokenIdx * pc.qStride + pc.qOffset;
    if (tid < pc.headDim) {
        uint gid = h * pc.headDim + tid;
        uint idxInHead = tid;
        if (idxInHead < pc.halfDim) {
            float cosVal = cosTable[tableBase + idxInHead];
            float sinVal = sinTable[tableBase + idxInHead];
            sharedQ[tid] = qIn[qRowBase + gid] * cosVal - qIn[qRowBase + gid + pc.halfDim] * sinVal;
        } else {
            uint pairIdx = idxInHead - pc.halfDim;
            float cosVal = cosTable[tableBase + pairIdx];
            float sinVal = sinTable[tableBase + pairIdx];
            sharedQ[tid] = qIn[qRowBase + gid - pc.halfDim] * sinVal + qIn[qRowBase + gid] * cosVal;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Compute attention scores = dot(Q_h, K_t) * scale
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        float dot = 0.0;
        uint kBase = t * pc.alignedKvDim + kvH * pc.headDim;
        for (uint d = 0; d < pc.headDim; d++) {
            dot += sharedQ[d] * float(kCache[kBase + d]);
        }
        sharedScores[t] = dot * pc.scale;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: Softmax over scores[0:seqLen]
    // 3a. Find max
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

    // 3b. Exp and sum
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

    // 3c. Normalize
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        sharedScores[t] /= sumExp;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 4: Weighted sum of V → output
    uint outStride = pc.nHeads * pc.headDim;
    if (tid < pc.headDim) {
        float acc = 0.0;
        for (uint t = 0; t < seqLen; t++) {
            acc += sharedScores[t] * float(vCache[t * pc.alignedKvDim + kvH * pc.headDim + tid]);
        }
        dataOut[tokenIdx * outStride + h * pc.headDim + tid] = acc;
    }
}
