// Native CUDA counterpart of MatMulCoopMatTiled.comp: blocked WMMA GEMM.
// 64x64 block tile, BK=32, 128-thread (4-warp) workgroup, double-buffered fp16
// shared staging, each warp owns a 2x2 grid of 16x16 tensor-core accumulators.
// Tensor-core MMA via inline-PTX helpers (mma.h is unusable under NVRTC).
#include "MatMulCoopMatCommon.cuh"

#define CMT_BM 64u
#define CMT_BN 64u
#define CMT_BK 32u

// Stage the next K-tile of A (64x32) and B (32x64) from global into the `buf`
// half of the double-buffered shared arrays. All 128 threads participate.
// fp16 elements loaded directly (no packed-uint tricks). Out-of-range global
// elements are zero-filled so edge blocks stay correct.
__device__ __forceinline__ void cmtStageTile(half* sA, half* sB, uint buf, uint k0,
                                              uint tid, uint blockRow, uint blockCol,
                                              const CoopMatPush& pc,
                                              const half* __restrict__ dataA,
                                              const half* __restrict__ dataB) {
    uint baseA = buf * CMT_BM * CMT_BK;
    uint baseB = buf * CMT_BK * CMT_BN;

    // Stage A: 64 rows x 32 K = 2048 fp16, 128 threads -> 16 each.
    for (uint i = tid; i < CMT_BM * CMT_BK; i += 128u) {
        uint row = i / CMT_BK;
        uint col = i % CMT_BK;
        uint gRow = blockRow + row;
        uint gCol = k0 + col;
        half v = (gRow < pc.M && gCol < pc.K) ? dataA[gRow * pc.strideA + gCol] : (half)0.0f;
        sA[baseA + row * CMT_BK + col] = v;
    }

    // Stage B: 32 K x 64 N = 2048 fp16, 128 threads -> 16 each.
    for (uint i = tid; i < CMT_BK * CMT_BN; i += 128u) {
        uint row = i / CMT_BN;
        uint col = i % CMT_BN;
        uint gRow = k0 + row;
        uint gCol = blockCol + col;
        half v = (gRow < pc.K && gCol < pc.N) ? dataB[gRow * pc.strideB + gCol] : (half)0.0f;
        sB[baseB + row * CMT_BN + col] = v;
    }
}

extern "C" __global__ void cut_main(const half* __restrict__ dataA,
                                    const half* __restrict__ dataB,
                                    const float* __restrict__ dataD,
                                    float* __restrict__ dataC,
                                    CoopMatPush pc) {
    __shared__ __align__(16) half sA[2u * CMT_BM * CMT_BK];  // [2][64][32]
    __shared__ __align__(16) half sB[2u * CMT_BK * CMT_BN];  // [2][32][64]

    uint tid = threadIdx.x;      // 0..127
    uint sgId = tid >> 5;        // warp id 0..3
    int lane = (int)(tid & 31u);
    uint blockRow = blockIdx.y * CMT_BM;
    uint blockCol = blockIdx.x * CMT_BN;
    uint subRow = (sgId / 2u) * 32u;   // quadrant offset inside the block
    uint subCol = (sgId % 2u) * 32u;

    Acc16 acc00, acc01, acc10, acc11;
    zeroAcc16(acc00); zeroAcc16(acc01); zeroAcc16(acc10); zeroAcc16(acc11);

    uint buf = 0u;
    cmtStageTile(sA, sB, 0u, 0u, tid, blockRow, blockCol, pc, dataA, dataB);
    __syncthreads();

    for (uint k0 = 0; k0 < pc.K; k0 += CMT_BK) {
        uint next = k0 + CMT_BK;
        if (next < pc.K)
            cmtStageTile(sA, sB, buf ^ 1u, next, tid, blockRow, blockCol, pc, dataA, dataB);

        if (sgId < 4u) {
            uint baseA = buf * CMT_BM * CMT_BK;
            uint baseB = buf * CMT_BK * CMT_BN;
            #pragma unroll
            for (uint kk = 0; kk < CMT_BK; kk += 16u) {
                FragA16 a0, a1;
                cutLoadA16(a0, sA + baseA + (subRow + 0u)  * CMT_BK + kk, (int)CMT_BK, lane);
                cutLoadA16(a1, sA + baseA + (subRow + 16u) * CMT_BK + kk, (int)CMT_BK, lane);
                FragB16 b0, b1;
                cutLoadB16(b0, sB + baseB + kk * CMT_BN + subCol,        (int)CMT_BN, lane);
                cutLoadB16(b1, sB + baseB + kk * CMT_BN + subCol + 16u,  (int)CMT_BN, lane);
                cutMma16(acc00, a0, b0);
                cutMma16(acc01, a0, b1);
                cutMma16(acc10, a1, b0);
                cutMma16(acc11, a1, b1);
            }
        }
        __syncthreads();
        buf ^= 1u;
    }

    if (sgId < 4u) {
        uint r0 = blockRow + subRow, c0 = blockCol + subCol;
        if (r0 < pc.M && c0 < pc.N)
            cutStoreC16(dataC + r0 * pc.strideC + c0, (int)pc.strideC, acc00, lane);
        if (r0 < pc.M && c0 + 16u < pc.N)
            cutStoreC16(dataC + r0 * pc.strideC + c0 + 16u, (int)pc.strideC, acc01, lane);
        if (r0 + 16u < pc.M && c0 < pc.N)
            cutStoreC16(dataC + (r0 + 16u) * pc.strideC + c0, (int)pc.strideC, acc10, lane);
        if (r0 + 16u < pc.M && c0 + 16u < pc.N)
            cutStoreC16(dataC + (r0 + 16u) * pc.strideC + c0 + 16u, (int)pc.strideC, acc11, lane);
    }
}
