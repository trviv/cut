// Native CUDA counterpart of DiTAttention.shader — keep semantics in lockstep.
//
// Fused non-causal multi-head attention for DiT/LTX models (LTX): no KV cache,
// no RoPE, no masking, nKvHeads == nHeads. Online softmax streams over KV tiles;
// no [sq, skv] score matrix is materialized. Each workgroup handles Q_TILE query
// rows of one head so K/V global reads amortize 16x. float4/half4 vectorized
// global loads (cut_mk_f4 widens half4 -> float4). Output is always float4.
//
// Float32 and Float16 input variants have DISTINCT SPIR-V hashes (buffer element
// type differs), so no hash-aliasing constraint applies; the body is dtype-generic
// through CUT_VEC_DTYPE_INPUT + cut_mk_f4, matching the transpiled reference.
#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

#ifndef CUT_VEC_DTYPE_INPUT
#define CUT_VEC_DTYPE_INPUT float4
#endif
typedef CUT_VEC_DTYPE_INPUT cut_dit_vec;

#define WG_SIZE 256
#define KV_TILE 128
#define Q_TILE 16
#define ROW_LANES 16

struct PushConstants {
    uint sq;
    uint skv;
    uint nHeads;
    uint headDim;
    uint strideQ;
    uint strideKV;
    uint strideO;
    float scale;
};

extern "C" __global__ void cut_main(const cut_dit_vec* __restrict__ qIn,
                                    const cut_dit_vec* __restrict__ kIn,
                                    const cut_dit_vec* __restrict__ vIn,
                                    float4* __restrict__ dataOut,
                                    PushConstants pc) {
    uint qTile = blockIdx.x;
    uint h = blockIdx.y;
    uint tid = threadIdx.x;
    uint qBase = qTile * Q_TILE;
    uint hd4 = pc.headDim >> 2;

    // Uniform across the workgroup, so the early-out is barrier-safe.
    if (qBase >= pc.sq || h >= pc.nHeads) return;

    __shared__ float sQ[Q_TILE][128];
    __shared__ float sS[Q_TILE][KV_TILE];
    __shared__ float sAcc[Q_TILE][128];
    __shared__ float sRowM[Q_TILE];
    __shared__ float sRowL[Q_TILE];
    __shared__ float sRowCorr[Q_TILE];
    __shared__ float sRed[Q_TILE][ROW_LANES];

    // Phase 1: load Q rows for this tile into shared memory
    for (uint idx = tid; idx < Q_TILE * hd4; idx += WG_SIZE) {
        uint qq = idx / hd4;
        uint d4 = idx % hd4;
        uint qi = qBase + qq;
        float4 qv = (qi < pc.sq)
            ? cut_mk_f4(qIn[((qi * pc.strideQ + h * pc.headDim) >> 2) + d4])
            : cut_mk_f4(0.0f, 0.0f, 0.0f, 0.0f);
        sQ[qq][d4 * 4 + 0] = qv.x;
        sQ[qq][d4 * 4 + 1] = qv.y;
        sQ[qq][d4 * 4 + 2] = qv.z;
        sQ[qq][d4 * 4 + 3] = qv.w;
        sAcc[qq][d4 * 4 + 0] = 0.0f;
        sAcc[qq][d4 * 4 + 1] = 0.0f;
        sAcc[qq][d4 * 4 + 2] = 0.0f;
        sAcc[qq][d4 * 4 + 3] = 0.0f;
    }
    if (tid < Q_TILE) {
        sRowM[tid] = -1e30f;
        sRowL[tid] = 0.0f;
    }
    __syncthreads();

    // Phase 2: online softmax over KV tiles
    uint nTiles = (pc.skv + KV_TILE - 1) / KV_TILE;
    for (uint tileIdx = 0; tileIdx < nTiles; tileIdx++) {
        uint tileBase = tileIdx * KV_TILE;
        uint j = tileBase + tid;

        // a. scores — one KV column per thread, all Q_TILE rows at once with each
        //    K element loaded once into a register. The qq loops MUST be unrolled
        //    so `partial` stays in registers (dynamic indexing would spill it).
        if (tid < KV_TILE) {
            if (j < pc.skv) {
                float partial[Q_TILE];
                #pragma unroll
                for (uint qz = 0; qz < Q_TILE; qz++) partial[qz] = 0.0f;
                uint kBase4 = (j * pc.strideKV + h * pc.headDim) >> 2;
                for (uint i4 = 0; i4 < hd4; i4++) {
                    float4 kv = cut_mk_f4(kIn[kBase4 + i4]);
                    #pragma unroll
                    for (uint qq = 0; qq < Q_TILE; qq++) {
                        partial[qq] += sQ[qq][i4 * 4 + 0] * kv.x
                                     + sQ[qq][i4 * 4 + 1] * kv.y
                                     + sQ[qq][i4 * 4 + 2] * kv.z
                                     + sQ[qq][i4 * 4 + 3] * kv.w;
                    }
                }
                #pragma unroll
                for (uint qw = 0; qw < Q_TILE; qw++)
                    sS[qw][tid] = partial[qw] * pc.scale;
            } else {
                #pragma unroll
                for (uint qq = 0; qq < Q_TILE; qq++)
                    sS[qq][tid] = -1e30f;
            }
        }
        __syncthreads();

        // b. per-row max + rescale factor: 16 lanes cooperate per row
        {
            uint row = tid / ROW_LANES;
            uint lane = tid % ROW_LANES;
            float pm = -1e30f;
            for (uint jj = lane; jj < KV_TILE; jj += ROW_LANES)
                pm = fmaxf(pm, sS[row][jj]);
            sRed[row][lane] = pm;
        }
        __syncthreads();
        if (tid < Q_TILE) {
            float rowMax = -1e30f;
            #pragma unroll
            for (uint ln = 0; ln < ROW_LANES; ln++)
                rowMax = fmaxf(rowMax, sRed[tid][ln]);
            float newM = fmaxf(sRowM[tid], rowMax);
            sRowCorr[tid] = expf(sRowM[tid] - newM);
            sRowM[tid] = newM;
        }
        __syncthreads();

        // c. probabilities — every thread converts its column in all rows
        if (tid < KV_TILE) {
            bool valid = (j < pc.skv);
            #pragma unroll
            for (uint qq = 0; qq < Q_TILE; qq++)
                sS[qq][tid] = valid ? expf(sS[qq][tid] - sRowM[qq]) : 0.0f;
        }
        __syncthreads();

        // d. per-row sum + normalizer update: 16 lanes cooperate per row
        {
            uint row = tid / ROW_LANES;
            uint lane = tid % ROW_LANES;
            float ps = 0.0f;
            for (uint jj = lane; jj < KV_TILE; jj += ROW_LANES)
                ps += sS[row][jj];
            sRed[row][lane] = ps;
        }
        __syncthreads();
        if (tid < Q_TILE) {
            float rowSum = 0.0f;
            #pragma unroll
            for (uint ln = 0; ln < ROW_LANES; ln++)
                rowSum += sRed[tid][ln];
            sRowL[tid] = sRowL[tid] * sRowCorr[tid] + rowSum;
        }
        __syncthreads();

        // e. accumulate V, rescaling the previous accumulator
        uint tileLen = min(KV_TILE, pc.skv - tileBase);
        for (uint idx = tid; idx < Q_TILE * hd4; idx += WG_SIZE) {
            uint qq = idx / hd4;
            uint d4 = idx % hd4;
            float corr = sRowCorr[qq];
            float4 a = cut_mk_f4(sAcc[qq][d4 * 4 + 0], sAcc[qq][d4 * 4 + 1],
                                 sAcc[qq][d4 * 4 + 2], sAcc[qq][d4 * 4 + 3]) * corr;
            for (uint jj = 0; jj < tileLen; jj++) {
                float4 vv = cut_mk_f4(vIn[(((tileBase + jj) * pc.strideKV + h * pc.headDim) >> 2) + d4]);
                a += sS[qq][jj] * vv;
            }
            sAcc[qq][d4 * 4 + 0] = a.x;
            sAcc[qq][d4 * 4 + 1] = a.y;
            sAcc[qq][d4 * 4 + 2] = a.z;
            sAcc[qq][d4 * 4 + 3] = a.w;
        }
        __syncthreads();
    }

    // Phase 3: normalize and write output
    for (uint idx = tid; idx < Q_TILE * hd4; idx += WG_SIZE) {
        uint qq = idx / hd4;
        uint d4 = idx % hd4;
        uint qi = qBase + qq;
        if (qi < pc.sq) {
            float invL = 1.0f / sRowL[qq];
            dataOut[((qi * pc.strideO + h * pc.headDim) >> 2) + d4] =
                cut_mk_f4(sAcc[qq][d4 * 4 + 0], sAcc[qq][d4 * 4 + 1],
                          sAcc[qq][d4 * 4 + 2], sAcc[qq][d4 * 4 + 3]) * invL;
        }
    }
}
