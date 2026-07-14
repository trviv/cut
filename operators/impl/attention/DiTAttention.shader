#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 128
#define KV_TILE 128
#define Q_TILE 8
#define ROW_LANES 16   // WG_SIZE / Q_TILE lanes cooperate per row

// Fused non-causal multi-head attention for DiT models (LTX): no KV cache,
// no RoPE, no masking, nKvHeads == nHeads. Q/K/V are row-major activations
// with heads interleaved along the channel dim (head h = columns
// h*headDim .. (h+1)*headDim-1). Online softmax streams over KV tiles, so no
// [sq, skv] score matrix is ever materialized. Query tiling: each workgroup
// processes Q_TILE query rows (same head) so K/V global reads amortize 8x
// and every thread stays busy in the V-accumulate phase.
// Uses float4-typed buffer views for vectorized global-memory access.

// Output: dataOut[sq, nHeads * headDim]
// Dispatch: (ceil(sq / Q_TILE), nHeads, 1) workgroups.
struct PushConstants {
    uint sq;
    uint skv;
    uint nHeads;
    uint headDim;
    uint strideQ;   // floats per Q row (innermost dim aligned to 4)
    uint strideKV;  // floats per K/V row (aligned)
    uint strideO;   // floats per output row (aligned)
    float scale;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> qIn;
[[vk::binding(1, 0)]] StructuredBuffer<float4> kIn;
[[vk::binding(2, 0)]] StructuredBuffer<float4> vIn;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> dataOut;

groupshared float sQ[Q_TILE][128];     // query rows (headDim <= 128)
groupshared float sS[Q_TILE][KV_TILE]; // raw scores, then probabilities
groupshared float sAcc[Q_TILE][128];   // output accumulators
groupshared float sRowM[Q_TILE];       // per-row running max
groupshared float sRowL[Q_TILE];       // per-row running normalizer
groupshared float sRowCorr[Q_TILE];    // per-row rescale factor for this tile
groupshared float sRed[Q_TILE][ROW_LANES]; // row-reduction staging

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint qTile = Gid.x;
    uint h = Gid.y;
    uint tid = GTid.x;
    uint qBase = qTile * Q_TILE;
    uint hd4 = pc.headDim >> 2;

    // Uniform across the workgroup, so the early-out is barrier-safe.
    if (qBase >= pc.sq || h >= pc.nHeads) return;

    // Phase 1: load Q rows for this tile into shared memory
    for (uint idx = tid; idx < Q_TILE * hd4; idx += WG_SIZE) {
        uint qq = idx / hd4;
        uint d4 = idx % hd4;
        uint qi = qBase + qq;
        float4 qv = (qi < pc.sq)
            ? qIn[((qi * pc.strideQ + h * pc.headDim) >> 2) + d4]
            : float4(0.0, 0.0, 0.0, 0.0);
        sQ[qq][d4 * 4 + 0] = qv.x;
        sQ[qq][d4 * 4 + 1] = qv.y;
        sQ[qq][d4 * 4 + 2] = qv.z;
        sQ[qq][d4 * 4 + 3] = qv.w;
        sAcc[qq][d4 * 4 + 0] = 0.0;
        sAcc[qq][d4 * 4 + 1] = 0.0;
        sAcc[qq][d4 * 4 + 2] = 0.0;
        sAcc[qq][d4 * 4 + 3] = 0.0;
    }
    if (tid < Q_TILE) {
        sRowM[tid] = -1e30;
        sRowL[tid] = 0.0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: online softmax over KV tiles
    uint nTiles = (pc.skv + KV_TILE - 1) / KV_TILE;
    for (uint tileIdx = 0; tileIdx < nTiles; tileIdx++) {
        uint tileBase = tileIdx * KV_TILE;
        uint j = tileBase + tid;

        // a. scores — one KV column per thread, all Q_TILE rows at once with
        //    each K element loaded once into a register. The qq loops MUST be
        //    unrolled so `partial` stays in registers (dynamic indexing would
        //    spill it to local memory).
        if (j < pc.skv) {
            float partial[Q_TILE];
            [unroll] for (uint qz = 0; qz < Q_TILE; qz++) partial[qz] = 0.0;
            uint kBase4 = (j * pc.strideKV + h * pc.headDim) >> 2;
            for (uint i4 = 0; i4 < hd4; i4++) {
                float4 kv = kIn[kBase4 + i4];
                [unroll] for (uint qq = 0; qq < Q_TILE; qq++) {
                    partial[qq] += sQ[qq][i4 * 4 + 0] * kv.x
                                 + sQ[qq][i4 * 4 + 1] * kv.y
                                 + sQ[qq][i4 * 4 + 2] * kv.z
                                 + sQ[qq][i4 * 4 + 3] * kv.w;
                }
            }
            [unroll] for (uint qw = 0; qw < Q_TILE; qw++)
                sS[qw][tid] = partial[qw] * pc.scale;
        } else {
            [unroll] for (uint qq = 0; qq < Q_TILE; qq++)
                sS[qq][tid] = -1e30;
        }
        GroupMemoryBarrierWithGroupSync();

        // b. per-row max + rescale factor: 16 lanes cooperate per row
        {
            uint row = tid / ROW_LANES;
            uint lane = tid % ROW_LANES;
            float pm = -1e30;
            for (uint jj = lane; jj < KV_TILE; jj += ROW_LANES)
                pm = max(pm, sS[row][jj]);
            sRed[row][lane] = pm;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < Q_TILE) {
            float rowMax = -1e30;
            [unroll] for (uint ln = 0; ln < ROW_LANES; ln++)
                rowMax = max(rowMax, sRed[tid][ln]);
            float newM = max(sRowM[tid], rowMax);
            sRowCorr[tid] = exp(sRowM[tid] - newM);
            sRowM[tid] = newM;
        }
        GroupMemoryBarrierWithGroupSync();

        // c. probabilities — every thread converts its column in all rows
        bool valid = (j < pc.skv);
        [unroll] for (uint qq = 0; qq < Q_TILE; qq++)
            sS[qq][tid] = valid ? exp(sS[qq][tid] - sRowM[qq]) : 0.0;
        GroupMemoryBarrierWithGroupSync();

        // d. per-row sum + normalizer update: 16 lanes cooperate per row
        {
            uint row = tid / ROW_LANES;
            uint lane = tid % ROW_LANES;
            float ps = 0.0;
            for (uint jj = lane; jj < KV_TILE; jj += ROW_LANES)
                ps += sS[row][jj];
            sRed[row][lane] = ps;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < Q_TILE) {
            float rowSum = 0.0;
            [unroll] for (uint ln = 0; ln < ROW_LANES; ln++)
                rowSum += sRed[tid][ln];
            sRowL[tid] = sRowL[tid] * sRowCorr[tid] + rowSum;
        }
        GroupMemoryBarrierWithGroupSync();

        // e. accumulate V, rescaling the previous accumulator
        uint tileLen = min(KV_TILE, pc.skv - tileBase);
        for (uint idx = tid; idx < Q_TILE * hd4; idx += WG_SIZE) {
            uint qq = idx / hd4;
            uint d4 = idx % hd4;
            float corr = sRowCorr[qq];
            float4 a = float4(sAcc[qq][d4 * 4 + 0], sAcc[qq][d4 * 4 + 1],
                              sAcc[qq][d4 * 4 + 2], sAcc[qq][d4 * 4 + 3]) * corr;
            for (uint jj = 0; jj < tileLen; jj++) {
                float4 vv = vIn[(((tileBase + jj) * pc.strideKV + h * pc.headDim) >> 2) + d4];
                a += sS[qq][jj] * vv;
            }
            sAcc[qq][d4 * 4 + 0] = a.x;
            sAcc[qq][d4 * 4 + 1] = a.y;
            sAcc[qq][d4 * 4 + 2] = a.z;
            sAcc[qq][d4 * 4 + 3] = a.w;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Phase 3: normalize and write output
    for (uint idx = tid; idx < Q_TILE * hd4; idx += WG_SIZE) {
        uint qq = idx / hd4;
        uint d4 = idx % hd4;
        uint qi = qBase + qq;
        if (qi < pc.sq) {
            float invL = 1.0 / sRowL[qq];
            dataOut[((qi * pc.strideO + h * pc.headDim) >> 2) + d4] =
                float4(sAcc[qq][d4 * 4 + 0], sAcc[qq][d4 * 4 + 1],
                       sAcc[qq][d4 * 4 + 2], sAcc[qq][d4 * 4 + 3]) * invL;
        }
    }
}
