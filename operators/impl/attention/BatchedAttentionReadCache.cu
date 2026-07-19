// Native CUDA counterpart of BatchedAttentionReadCache.shader — keep
// semantics in lockstep.
//
// Batched attention reading a pre-populated K/V cache (see
// BatchedKVCacheWrite); Q gets RoPE inline. One workgroup per (head, token);
// threads collaborate on the per-token softmax over seqLen = pos+1 cache
// positions. Output: dataOut[N, nHeads * headDim]. Dispatch:
// (nHeads, batchSize, 1).
#include "ComputeOpsShared.h"
#include "AttentionCommon.cuh"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif
#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 2048
#endif

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

extern "C" __global__ void cut_main(const float* __restrict__ qIn,
                                    const cut_kv_t* __restrict__ kCache,
                                    const cut_kv_t* __restrict__ vCache,
                                    const uint* __restrict__ positions,
                                    const float* __restrict__ cosTable,
                                    const float* __restrict__ sinTable,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint h = blockIdx.x;
    uint tokenIdx = blockIdx.y;
    uint tid = threadIdx.x;
    uint kvH = h / pc.nRep;

    // Uniform across the workgroup, so the early-out is barrier-safe.
    if (tokenIdx >= pc.batchSize) return;

    uint pos = positions[tokenIdx];
    uint seqLen = pos + 1;
    uint tableBase = pos * pc.halfDim;

    __shared__ float sharedQ[128];
    __shared__ float sharedScores[MAX_SEQ_LEN];
    __shared__ float red[CUT_ATTN_NWARPS];

    // Phase 1: RoPE(Q) for this head into shared memory
    uint qRowBase = tokenIdx * pc.qStride + pc.qOffset;
    if (tid < pc.headDim) {
        uint gid = h * pc.headDim + tid;
        sharedQ[tid] = cut_rope_half_split(qIn, qRowBase, gid, cosTable, sinTable,
                                           tableBase, pc.headDim, pc.halfDim);
    }
    __syncthreads();

    // Phase 2: scores = dot(Q_h, K_t) * scale  for t in 0..seqLen-1
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        float d = 0.0f;
        uint kBase = t * pc.alignedKvDim + kvH * pc.headDim;
        for (uint i = 0; i < pc.headDim; i++) {
            d += sharedQ[i] * cut_kv_load(kCache[kBase + i]);
        }
        sharedScores[t] = d * pc.scale;
    }
    __syncthreads();

    // Phase 3: softmax (warp-shuffle block reductions; all 256 threads call)
    float localMax = -1e30f;
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        localMax = fmaxf(localMax, sharedScores[t]);
    }
    float maxScore = cut_block_reduce_max(localMax, red, tid);

    float localSum = 0.0f;
    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        sharedScores[t] = expf(sharedScores[t] - maxScore);
        localSum += sharedScores[t];
    }
    float sumExp = cut_block_reduce_sum(localSum, red, tid);

    for (uint t = tid; t < seqLen; t += WG_SIZE) {
        sharedScores[t] /= sumExp;
    }
    __syncthreads();

    // Phase 4: weighted sum of V
    uint outStride = pc.nHeads * pc.headDim;
    if (tid < pc.headDim) {
        float acc = 0.0f;
        for (uint t = 0; t < seqLen; t++) {
            acc += sharedScores[t]
                 * cut_kv_load(vCache[t * pc.alignedKvDim + kvH * pc.headDim + tid]);
        }
        dataOut[tokenIdx * outStride + h * pc.headDim + tid] = acc;
    }
}
