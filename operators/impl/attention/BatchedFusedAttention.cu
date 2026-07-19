// Native CUDA counterpart of BatchedFusedAttention.shader — keep semantics in
// lockstep.
//
// Batched fused RoPE + CacheWrite + Attention for prefill (N tokens). Head 0
// of each token writes RoPE(K)/V to the cache row for that token's position,
// then every (head, token) workgroup runs RoPE(Q) + scores + softmax +
// weighted V sum over seqLen = pos+1 positions.
//
// NOTE: like the shader, this kernel has a cross-workgroup race (token j's
// attention can read token i's cache row before token i's head-0 write
// completes); prefer BatchedKVCacheWrite + BatchedAttentionReadCache. The
// race is mirrored, not fixed — semantics stay in lockstep with the shader.
#include "ComputeOpsShared.h"
#include "AttentionCommon.cuh"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif
#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 2048
#endif

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

extern "C" __global__ void cut_main(const float* __restrict__ qIn,
                                    const float* __restrict__ kIn,
                                    const float* __restrict__ vIn,
                                    cut_kv_t* __restrict__ kCache,
                                    cut_kv_t* __restrict__ vCache,
                                    const uint* __restrict__ posBuffer,
                                    const float* __restrict__ cosTable,
                                    const float* __restrict__ sinTable,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint h = blockIdx.x;        // Attention head index
    uint tokenIdx = blockIdx.y; // Which token in the batch
    uint tid = threadIdx.x;     // Thread within workgroup
    uint kvH = h / pc.nRep;     // KV head index (GQA)

    // Uniform across the workgroup, so the early-out is barrier-safe.
    if (tokenIdx >= pc.batchSize) return;

    uint pos = posBuffer[tokenIdx];
    uint seqLen = pos + 1; // Causal: attend to positions [0..pos]
    uint tableBase = pos * pc.halfDim;

    __shared__ float sharedQ[128]; // Max head_dim
    __shared__ float sharedScores[MAX_SEQ_LEN];
    __shared__ float red[CUT_ATTN_NWARPS];

    // Phase 0: First head writes RoPE(K) and V to cache at this token's position
    if (h == 0) {
        uint kRowBase = tokenIdx * pc.kStride + pc.kOffset;
        uint vRowBase = tokenIdx * pc.vStride + pc.vOffset;

        // Write RoPE(K) to K cache
        for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
            float kVal = cut_rope_half_split(kIn, kRowBase, d, cosTable, sinTable,
                                             tableBase, pc.headDim, pc.halfDim);
            kCache[pos * pc.alignedKvDim + d] = cut_kv_store(kVal);
        }

        // Write V to V cache (no RoPE)
        for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
            vCache[pos * pc.alignedKvDim + d] = cut_kv_store(vIn[vRowBase + d]);
        }
    }

    // All heads must wait for cache write by head 0 of THIS token.
    // Since different tokens write to different cache positions, no cross-token
    // conflict. But heads within a token share the cache row.
    __syncthreads();

    // Phase 1: Load RoPE(Q) for this head into shared memory
    uint qRowBase = tokenIdx * pc.qStride + pc.qOffset;
    if (tid < pc.headDim) {
        uint gid = h * pc.headDim + tid;
        sharedQ[tid] = cut_rope_half_split(qIn, qRowBase, gid, cosTable, sinTable,
                                           tableBase, pc.headDim, pc.halfDim);
    }
    __syncthreads();

    // Phase 2: Compute attention scores = dot(Q_h, K_t) * scale
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        float dot = 0.0f;
        uint kBase = t * pc.alignedKvDim + kvH * pc.headDim;
        for (uint d = 0; d < pc.headDim; d++) {
            dot += sharedQ[d] * cut_kv_load(kCache[kBase + d]);
        }
        sharedScores[t] = dot * pc.scale;
    }
    __syncthreads();

    // Phase 3: Softmax over scores[0:seqLen]
    // 3a. Find max (warp-shuffle block reduction; all 256 threads call)
    float localMax = -1e30f;
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        localMax = fmaxf(localMax, sharedScores[t]);
    }
    float maxScore = cut_block_reduce_max(localMax, red, tid);

    // 3b. Exp and sum
    float localSum = 0.0f;
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        sharedScores[t] = expf(sharedScores[t] - maxScore);
        localSum += sharedScores[t];
    }
    float sumExp = cut_block_reduce_sum(localSum, red, tid);

    // 3c. Normalize
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        sharedScores[t] /= sumExp;
    }
    __syncthreads();

    // Phase 4: Weighted sum of V → output
    uint outStride = pc.nHeads * pc.headDim;
    if (tid < pc.headDim) {
        float acc = 0.0f;
        for (uint t = 0; t < seqLen; t++) {
            acc += sharedScores[t] * cut_kv_load(vCache[t * pc.alignedKvDim + kvH * pc.headDim + tid]);
        }
        dataOut[tokenIdx * outStride + h * pc.headDim + tid] = acc;
    }
}
