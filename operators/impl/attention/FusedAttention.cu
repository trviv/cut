// Native CUDA counterpart of FusedAttention.shader — keep semantics in
// lockstep.
//
// Fused RoPE + CacheWrite + Attention for single-token decode (saves 4
// dispatches/layer): workgroup 0 writes RoPE(K) and V into the caches, every
// workgroup (one per head) applies RoPE to its Q, then scores + softmax +
// weighted V sum over seqLen cache positions.
#include "ComputeOpsShared.h"
#include "AttentionCommon.cuh"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif
#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 2048
#endif

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

extern "C" __global__ void cut_main(const float* __restrict__ qIn,
                                    const float* __restrict__ kIn,
                                    const float* __restrict__ vIn,
                                    cut_kv_t* __restrict__ kCache,
                                    cut_kv_t* __restrict__ vCache,
                                    const uint* __restrict__ runtimeParams,
                                    const float* __restrict__ cosTable,
                                    const float* __restrict__ sinTable,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint h = blockIdx.x;    // Attention head index
    uint tid = threadIdx.x; // Thread within workgroup
    uint kvH = h / pc.nRep; // KV head index (GQA)
    uint pos = runtimeParams[0];
    uint seqLen = runtimeParams[1];
    uint tableBase = pos * pc.halfDim;

    __shared__ float sharedQ[128]; // Max head_dim (post-RoPE Q for this head)
    __shared__ float sharedScores[MAX_SEQ_LEN];
    __shared__ float red[CUT_ATTN_NWARPS];

    // Phase 0: First workgroup writes RoPE(K) and V to cache
    // All workgroups need the updated cache, so only the first WG does the
    // write. Other WGs wait via the seqLen parameter (seqLen = pos+1 includes
    // this token).
    if (h == 0) {
        // Write RoPE(K) to K cache at position pos
        for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
            float kVal = cut_rope_half_split(kIn, 0, d, cosTable, sinTable,
                                             tableBase, pc.headDim, pc.halfDim);
            kCache[pos * pc.alignedKvDim + d] = cut_kv_store(kVal);
        }

        // Write V to V cache at position pos (no RoPE needed for V)
        for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
            vCache[pos * pc.alignedKvDim + d] = cut_kv_store(vIn[d]);
        }
    }

    // Phase 1: Load RoPE(Q) for this head into shared memory
    if (tid < pc.headDim) {
        uint gid = h * pc.headDim + tid;
        sharedQ[tid] = cut_rope_half_split(qIn, 0, gid, cosTable, sinTable,
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

    // Phase 4: Weighted sum of V
    if (tid < pc.headDim) {
        float acc = 0.0f;
        for (uint t = 0; t < seqLen; t++) {
            acc += sharedScores[t] * cut_kv_load(vCache[t * pc.alignedKvDim + kvH * pc.headDim + tid]);
        }
        dataOut[h * pc.headDim + tid] = acc;
    }
}
