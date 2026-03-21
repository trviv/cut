#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256
#define MAX_SEQ_LEN 2048

struct PushConstants {
    uint nHeads;
    uint nKvHeads;
    uint headDim;
    uint kvDim;        // nKvHeads * headDim
    uint alignedKvDim; // (kvDim + 3) & ~3
    uint nRep;         // nHeads / nKvHeads
    float scale;       // 1/sqrt(headDim)
    uint halfDim;      // headDim / 2 for RoPE
};
[[vk::push_constant]] PushConstants pc;

// Inputs: Q [nHeads * headDim] (pre-RoPE) — always float
[[vk::binding(0, 0)]] StructuredBuffer<float> qIn;
// K [kvDim] (pre-RoPE) — always float
[[vk::binding(1, 0)]] StructuredBuffer<float> kIn;
// V [kvDim] — always float
[[vk::binding(2, 0)]] StructuredBuffer<float> vIn;
// K cache [maxSeqLen, alignedKvDim] — float or half
#ifdef DTYPE_INPUT_IS_HALF
[[vk::binding(3, 0)]] RWStructuredBuffer<float16_t> kCache;
[[vk::binding(4, 0)]] RWStructuredBuffer<float16_t> vCache;
#else
[[vk::binding(3, 0)]] RWStructuredBuffer<float> kCache;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> vCache;
#endif
// Runtime params [pos, seqLen]
[[vk::binding(5, 0)]] StructuredBuffer<uint> runtimeParams;
// Precomputed cos table [maxSeqLen * halfDim]
[[vk::binding(6, 0)]] StructuredBuffer<float> cosTable;
// Precomputed sin table [maxSeqLen * halfDim]
[[vk::binding(7, 0)]] StructuredBuffer<float> sinTable;
// Output [nHeads * headDim] — always float
[[vk::binding(8, 0)]] RWStructuredBuffer<float> dataOut;

// Shared memory — always float for numerical stability
groupshared float sharedQ[128];            // Max head_dim (post-RoPE Q for this head)
groupshared float sharedScores[MAX_SEQ_LEN];
groupshared float sharedReduce[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID,
          uint3 GTid : SV_GroupThreadID,
          uint3 Gid  : SV_GroupID) {
    uint h = Gid.x;      // Attention head index
    uint tid = GTid.x;   // Thread within workgroup
    uint kvH = h / pc.nRep; // KV head index (GQA)
    uint pos = runtimeParams[0];
    uint seqLen = runtimeParams[1];
    uint tableBase = pos * pc.halfDim;

    // Phase 0: First workgroup writes RoPE(K) and V to cache
    // All workgroups need the updated cache, so only the first WG does the write.
    // Other WGs wait via the seqLen parameter (seqLen = pos+1 includes this token).
    if (h == 0) {
        // Write RoPE(K) to K cache at position pos
        for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
            uint idxInHead = d % pc.headDim;
            float kVal;
            if (idxInHead < pc.halfDim) {
                float cosVal = cosTable[tableBase + idxInHead];
                float sinVal = sinTable[tableBase + idxInHead];
                kVal = kIn[d] * cosVal - kIn[d + pc.halfDim] * sinVal;
            } else {
                uint pairIdx = idxInHead - pc.halfDim;
                float cosVal = cosTable[tableBase + pairIdx];
                float sinVal = sinTable[tableBase + pairIdx];
                kVal = kIn[d - pc.halfDim] * sinVal + kIn[d] * cosVal;
            }
#ifdef DTYPE_INPUT_IS_HALF
            kCache[pos * pc.alignedKvDim + d] = float16_t(kVal);
#else
            kCache[pos * pc.alignedKvDim + d] = kVal;
#endif
        }

        // Write V to V cache at position pos (no RoPE needed for V)
        for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
#ifdef DTYPE_INPUT_IS_HALF
            vCache[pos * pc.alignedKvDim + d] = float16_t(vIn[d]);
#else
            vCache[pos * pc.alignedKvDim + d] = vIn[d];
#endif
        }
    }

    // Phase 1: Load RoPE(Q) for this head into shared memory
    if (tid < pc.headDim) {
        uint gid = h * pc.headDim + tid;
        uint idxInHead = tid;
        if (idxInHead < pc.halfDim) {
            float cosVal = cosTable[tableBase + idxInHead];
            float sinVal = sinTable[tableBase + idxInHead];
            sharedQ[tid] = qIn[gid] * cosVal - qIn[gid + pc.halfDim] * sinVal;
        } else {
            uint pairIdx = idxInHead - pc.halfDim;
            float cosVal = cosTable[tableBase + pairIdx];
            float sinVal = sinTable[tableBase + pairIdx];
            sharedQ[tid] = qIn[gid - pc.halfDim] * sinVal + qIn[gid] * cosVal;
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

    // Phase 4: Weighted sum of V
    if (tid < pc.headDim) {
        float acc = 0.0;
        for (uint t = 0; t < seqLen; t++) {
            acc += sharedScores[t] * float(vCache[t * pc.alignedKvDim + kvH * pc.headDim + tid]);
        }
        dataOut[h * pc.headDim + tid] = acc;
    }
}
