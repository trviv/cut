// Native CUDA counterpart of Attention.shader — keep semantics in lockstep.
//
// Scaled dot-product attention with GQA support: one workgroup per head,
// scores + softmax over the populated KV cache, weighted V sum.
//
// NOTE: the Int8 and Float32 variants of this shader hash-alias (identical
// SPIR-V) and share one compiled kernel (built with the Float32 defines), so
// this code must not branch on any Int8 macro. Branching on
// CUT_DTYPE_INPUT_IS_HALF (via cut_kv_t) is fine: the Float16 variant has its
// own hash.
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
};

extern "C" __global__ void cut_main(const float* __restrict__ query,
                                    const cut_kv_t* __restrict__ kCache,
                                    const cut_kv_t* __restrict__ vCache,
                                    const uint* __restrict__ runtimeParams,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint h = blockIdx.x;    // Attention head index
    uint tid = threadIdx.x; // Thread within workgroup
    uint kvH = h / pc.nRep; // KV head index (GQA)
    uint seqLen = runtimeParams[1];

    __shared__ float sharedQ[128]; // Max head_dim
    __shared__ float sharedScores[MAX_SEQ_LEN];
    __shared__ float red[CUT_ATTN_NWARPS];

    // Phase 1: Load Q for this head into shared memory
    if (tid < pc.headDim) {
        sharedQ[tid] = query[h * pc.headDim + tid];
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
    // 3a. Find max (warp shuffle + 8-slot staging instead of the shader's
    //     shared-memory tree — same block geometry, all 256 threads call).
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
