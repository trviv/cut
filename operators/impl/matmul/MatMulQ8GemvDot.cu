// MatMulQ8GemvDot.cu
// Native CUDA counterpart of MatMulQ8GemvDot.comp (Q8 GEMV, on-the-fly int8 A quant + __dp4a).
#include "MatMulQ8Common.cuh"

static __device__ __forceinline__ uint cut_pack_i8x4(int x0, int x1, int x2, int x3) {
    return (uint(x0) & 0xFFu) | ((uint(x1) & 0xFFu) << 8) |
           ((uint(x2) & 0xFFu) << 16) | ((uint(x3) & 0xFFu) << 24);
}

static __device__ __forceinline__ float cut_warpMax(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

extern "C" __global__ void cut_main(
    const CUT_VEC_DTYPE_INPUT1* __restrict__ dataA,
    const uint* __restrict__ packedB,
    const CUT_VEC_DTYPE_SCALES* __restrict__ scalesB,
    const CUT_VEC_DTYPE_OUTPUT* __restrict__ dataD,
    CUT_SCALAR_DTYPE_OUTPUT* __restrict__ dataC,
    PushConstants pc) {

    uint tid = threadIdx.x;
    uint baseN = blockIdx.x * 4;
    uint m = blockIdx.y;

    if (baseN >= pc.N || m >= pc.M) return;

    uint group = tid >> 2;
    uint lane = tid & 3;
    uint col = baseN + lane;
    bool colValid = (col < pc.N);

    float acc = 0.0f;

    uint numBlocks = pc.K / 32;

    __shared__ float sA[32];
    __shared__ signed char sB[32][4];

    for (uint blk = 0; blk < numBlocks; blk++) {
        uint k_base = blk * 32;

        sA[tid] = (float)cut_loadA_fast(dataA, pc, m, k_base + tid);

        uint byteIdx = (k_base + tid) * pc.strideBN + baseN;
        uint packed = packedB[byteIdx >> 2];
        int sp = (int)packed;
        sB[tid][0] = (signed char)((sp << 24) >> 24);
        sB[tid][1] = (signed char)((sp << 16) >> 24);
        sB[tid][2] = (signed char)((sp << 8) >> 24);
        sB[tid][3] = (signed char)(sp >> 24);

        __syncthreads();

        float a_abs = fabsf(sA[tid]);
        float a_max = cut_warpMax(a_abs);
        float a_scale = a_max / 127.0f;
        float a_scale_inv = (a_max > 0.0f) ? (127.0f / a_max) : 0.0f;
        int a_qi = clamp((int)round(sA[tid] * a_scale_inv), -127, 127);

        int aq0 = __shfl_sync(0xffffffffu, a_qi, group * 4 + 0);
        int aq1 = __shfl_sync(0xffffffffu, a_qi, group * 4 + 1);
        int aq2 = __shfl_sync(0xffffffffu, a_qi, group * 4 + 2);
        int aq3 = __shfl_sync(0xffffffffu, a_qi, group * 4 + 3);
        uint a_packed = cut_pack_i8x4(aq0, aq1, aq2, aq3);

        int bv0 = sB[group * 4 + 0][lane];
        int bv1 = sB[group * 4 + 1][lane];
        int bv2 = sB[group * 4 + 2][lane];
        int bv3 = sB[group * 4 + 3][lane];
        uint b_packed = cut_pack_i8x4(bv0, bv1, bv2, bv3);

        int dot = __dp4a((int)a_packed, (int)b_packed, 0);

        float b_scale = 0.0f;
        if (colValid) {
            uint sidx = blk * pc.scaleStride + col;
            b_scale = float(scalesB[sidx >> 2][sidx & 3]);
        }
        acc += (float)dot * a_scale * b_scale;

        __syncthreads();
    }

    uint partialK = pc.K - numBlocks * 32;
    if (partialK > 0) {
        uint k_base = numBlocks * 32;

        sA[tid] = (tid < partialK) ? (float)cut_loadA_fast(dataA, pc, m, k_base + tid) : 0.0f;

        if (tid < partialK) {
            uint byteIdx = (k_base + tid) * pc.strideBN + baseN;
            uint packed = packedB[byteIdx >> 2];
            int sp = (int)packed;
            sB[tid][0] = (signed char)((sp << 24) >> 24);
            sB[tid][1] = (signed char)((sp << 16) >> 24);
            sB[tid][2] = (signed char)((sp << 8) >> 24);
            sB[tid][3] = (signed char)(sp >> 24);
        } else {
            sB[tid][0] = 0;
            sB[tid][1] = 0;
            sB[tid][2] = 0;
            sB[tid][3] = 0;
        }

        __syncthreads();

        float a_abs = (tid < partialK) ? fabsf(sA[tid]) : 0.0f;
        float a_max = cut_warpMax(a_abs);
        float a_scale = a_max / 127.0f;
        float a_scale_inv = (a_max > 0.0f) ? (127.0f / a_max) : 0.0f;
        int a_qi = (tid < partialK) ? clamp((int)round(sA[tid] * a_scale_inv), -127, 127) : 0;

        uint k0 = k_base + group * 4;
        bool groupValid = (k0 < pc.K);

        int aq0 = __shfl_sync(0xffffffffu, a_qi, group * 4 + 0);
        int aq1 = __shfl_sync(0xffffffffu, a_qi, group * 4 + 1);
        int aq2 = __shfl_sync(0xffffffffu, a_qi, group * 4 + 2);
        int aq3 = __shfl_sync(0xffffffffu, a_qi, group * 4 + 3);
        uint a_packed = cut_pack_i8x4(aq0, aq1, aq2, aq3);

        int bv0 = sB[group * 4 + 0][lane];
        int bv1 = sB[group * 4 + 1][lane];
        int bv2 = sB[group * 4 + 2][lane];
        int bv3 = sB[group * 4 + 3][lane];
        uint b_packed = cut_pack_i8x4(bv0, bv1, bv2, bv3);

        int dot = __dp4a((int)a_packed, (int)b_packed, 0);

        uint sidx = numBlocks * pc.scaleStride + col;
        float b_scale = (colValid && groupValid) ? float(scalesB[sidx >> 2][sidx & 3]) : 0.0f;
        acc += (float)dot * a_scale * b_scale;

        __syncthreads();
    }

    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 16);

    if (tid < 4 && colValid) {
        cut_writeOutput(dataC, dataD, pc, m, col, acc);
    }
}
