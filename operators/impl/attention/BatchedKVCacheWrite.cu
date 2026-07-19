// Native CUDA counterpart of BatchedKVCacheWrite.shader — keep semantics in
// lockstep.
//
// Batched K + V cache write for prefill, with K-side RoPE:
//   kCache[positions[i], :] = RoPE(kIn[i, kOffset : kOffset + kvDim])
//   vCache[positions[i], :] =        vIn[i, vOffset : vOffset + kvDim]
// Dispatch: (1, N, 1) — one workgroup per token.
#include "ComputeOpsShared.h"
#include "AttentionCommon.cuh"

#define WG_SIZE 256

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

extern "C" __global__ void cut_main(const float* __restrict__ kIn,
                                    const float* __restrict__ vIn,
                                    cut_kv_t* __restrict__ kCache,
                                    cut_kv_t* __restrict__ vCache,
                                    const uint* __restrict__ positions,
                                    const float* __restrict__ cosTable,
                                    const float* __restrict__ sinTable,
                                    PushConstants pc) {
    uint tokenIdx = blockIdx.y;
    uint tid = threadIdx.x;
    if (tokenIdx >= pc.batchSize) return;

    uint pos = positions[tokenIdx];
    uint tableBase = pos * pc.halfDim;
    uint kRowBase = tokenIdx * pc.kStride + pc.kOffset;
    uint vRowBase = tokenIdx * pc.vStride + pc.vOffset;
    uint cacheBase = pos * pc.alignedKvDim;

    // Write RoPE(K) to K cache
    for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
        float kVal = cut_rope_half_split(kIn, kRowBase, d, cosTable, sinTable,
                                         tableBase, pc.headDim, pc.halfDim);
        kCache[cacheBase + d] = cut_kv_store(kVal);
    }

    // Write V to V cache (no RoPE)
    for (uint d = tid; d < pc.kvDim; d += WG_SIZE) {
        vCache[cacheBase + d] = cut_kv_store(vIn[vRowBase + d]);
    }
}
