// MatMulQ8TiledDot.cu
// Native CUDA counterpart of MatMulQ8TiledDot.comp (Q8 tiled matmul, on-the-fly int8 A quant + __dp4a).

#include "MatMulQ8Common.cuh"

static __device__ __forceinline__ uint cut_pack_i8x4(int x0, int x1, int x2, int x3) {
    return (uint(x0) & 0xFFu) | ((uint(x1) & 0xFFu) << 8) |
           ((uint(x2) & 0xFFu) << 16) | ((uint(x3) & 0xFFu) << 24);
}

extern "C" __global__ void cut_main(const CUT_VEC_DTYPE_INPUT1* __restrict__ dataA,
    const uint* __restrict__ packedB, const CUT_VEC_DTYPE_SCALES* __restrict__ scalesB,
    const CUT_VEC_DTYPE_OUTPUT* __restrict__ dataD, CUT_SCALAR_DTYPE_OUTPUT* __restrict__ dataC,
    PushConstants pc) {
    uint tid = threadIdx.x;
    uint tileN = blockIdx.x * 64u;
    uint tileM = blockIdx.y * 32u;

    if (tileN >= pc.N || tileM >= pc.M) return;

    uint a_row     = tid >> 3;
    uint a_k_local = (tid & 7u) << 2;
    uint b_row     = tid >> 3;
    uint b_col     = (tid & 7u) << 3;
    uint localRow  = tid >> 4;
    uint localCol  = tid & 15u;

    __shared__ signed char tileA[32][32];
    __shared__ signed char tileB[32][64];
    __shared__ float a_scales[32];
    __shared__ float b_scales[64];

    float acc[2][4];
    for (uint m = 0; m < 2u; m++)
        for (uint n = 0; n < 4u; n++)
            acc[m][n] = 0.0f;

    uint numBlocks = pc.K / 32u;

    for (uint blk = 0u; blk < numBlocks; blk++) {
        uint k_base = blk * 32u;

        // STEP 1: Load B int8 tile
        {
            bool kValid = (k_base + b_row < pc.K);
            uint byteIdx = (k_base + b_row) * pc.strideBN + tileN + b_col;
            uint p0 = (kValid && tileN + b_col < pc.N) ? packedB[byteIdx >> 2] : 0u;
            uint p1 = (kValid && tileN + b_col + 4u < pc.N) ? packedB[(byteIdx + 4u) >> 2] : 0u;
            int sp0 = (int)p0;
            int sp1 = (int)p1;
            tileB[b_row][b_col + 0u] = (signed char)((sp0 << 24) >> 24);
            tileB[b_row][b_col + 1u] = (signed char)((sp0 << 16) >> 24);
            tileB[b_row][b_col + 2u] = (signed char)((sp0 <<  8) >> 24);
            tileB[b_row][b_col + 3u] = (signed char)(sp0 >> 24);
            tileB[b_row][b_col + 4u] = (signed char)((sp1 << 24) >> 24);
            tileB[b_row][b_col + 5u] = (signed char)((sp1 << 16) >> 24);
            tileB[b_row][b_col + 6u] = (signed char)((sp1 <<  8) >> 24);
            tileB[b_row][b_col + 7u] = (signed char)(sp1 >> 24);
        }

        // STEP 2: Load B scales
        if (tid < 64u) {
            uint sidx = blk * pc.scaleStride + tileN + tid;
            b_scales[tid] = (tileN + tid < pc.N) ? float(scalesB[sidx >> 2][sidx & 3]) : 0.0f;
        }

        // STEP 3: Load A float, quantize per-row to int8
        {
            bool rowValid = (tileM + a_row < pc.M);
            float a0 = rowValid ? (float)cut_loadA_fast(dataA, pc, tileM + a_row, k_base + a_k_local + 0u) : 0.0f;
            float a1 = rowValid ? (float)cut_loadA_fast(dataA, pc, tileM + a_row, k_base + a_k_local + 1u) : 0.0f;
            float a2 = rowValid ? (float)cut_loadA_fast(dataA, pc, tileM + a_row, k_base + a_k_local + 2u) : 0.0f;
            float a3 = rowValid ? (float)cut_loadA_fast(dataA, pc, tileM + a_row, k_base + a_k_local + 3u) : 0.0f;

            float lm = fmaxf(fmaxf(fabsf(a0), fabsf(a1)), fmaxf(fabsf(a2), fabsf(a3)));
            lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, 4u));
            lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, 2u));
            lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, 1u));

            float a_scale_inv = (lm > 0.0f) ? (127.0f / lm) : 0.0f;
            if ((tid & 7u) == 0u) a_scales[a_row] = lm / 127.0f;

            tileA[a_row][a_k_local + 0u] = (signed char)clamp((int)round(a0 * a_scale_inv), -127, 127);
            tileA[a_row][a_k_local + 1u] = (signed char)clamp((int)round(a1 * a_scale_inv), -127, 127);
            tileA[a_row][a_k_local + 2u] = (signed char)clamp((int)round(a2 * a_scale_inv), -127, 127);
            tileA[a_row][a_k_local + 3u] = (signed char)clamp((int)round(a3 * a_scale_inv), -127, 127);
        }

        __syncthreads();

        // STEP 4: Compute integer dot products for this K-tile
        {
            int acc_int[2][4];
            for (uint m = 0; m < 2u; m++)
                for (uint n = 0; n < 4u; n++)
                    acc_int[m][n] = 0;

            for (uint k = 0u; k < 32u; k += 4u) {
                uint a_p0 = cut_pack_i8x4(tileA[localRow][k],     tileA[localRow][k+1u],     tileA[localRow][k+2u],     tileA[localRow][k+3u]);
                uint a_p1 = cut_pack_i8x4(tileA[localRow+16u][k], tileA[localRow+16u][k+1u], tileA[localRow+16u][k+2u], tileA[localRow+16u][k+3u]);
                uint b_p0 = cut_pack_i8x4(tileB[k][localCol],     tileB[k+1u][localCol],     tileB[k+2u][localCol],     tileB[k+3u][localCol]);
                uint b_p1 = cut_pack_i8x4(tileB[k][localCol+16u], tileB[k+1u][localCol+16u], tileB[k+2u][localCol+16u], tileB[k+3u][localCol+16u]);
                uint b_p2 = cut_pack_i8x4(tileB[k][localCol+32u], tileB[k+1u][localCol+32u], tileB[k+2u][localCol+32u], tileB[k+3u][localCol+32u]);
                uint b_p3 = cut_pack_i8x4(tileB[k][localCol+48u], tileB[k+1u][localCol+48u], tileB[k+2u][localCol+48u], tileB[k+3u][localCol+48u]);

                acc_int[0][0] = __dp4a((int)a_p0, (int)b_p0, acc_int[0][0]);
                acc_int[0][1] = __dp4a((int)a_p0, (int)b_p1, acc_int[0][1]);
                acc_int[0][2] = __dp4a((int)a_p0, (int)b_p2, acc_int[0][2]);
                acc_int[0][3] = __dp4a((int)a_p0, (int)b_p3, acc_int[0][3]);
                acc_int[1][0] = __dp4a((int)a_p1, (int)b_p0, acc_int[1][0]);
                acc_int[1][1] = __dp4a((int)a_p1, (int)b_p1, acc_int[1][1]);
                acc_int[1][2] = __dp4a((int)a_p1, (int)b_p2, acc_int[1][2]);
                acc_int[1][3] = __dp4a((int)a_p1, (int)b_p3, acc_int[1][3]);
            }

            for (uint m = 0; m < 2u; m++) {
                float as = a_scales[localRow + m * 16u];
                for (uint n = 0; n < 4u; n++) {
                    float bs = b_scales[localCol + n * 16u];
                    acc[m][n] += (float)acc_int[m][n] * as * bs;
                }
            }
        }

        __syncthreads();
    }

    // Handle partial K block
    {
        uint partialK = pc.K - numBlocks * 32u;
        if (partialK > 0u) {
            uint k_base = numBlocks * 32u;

            // STEP 1: Load B int8 tile
            {
                bool kValid = (k_base + b_row < pc.K);
                uint byteIdx = (k_base + b_row) * pc.strideBN + tileN + b_col;
                uint p0 = (kValid && tileN + b_col < pc.N) ? packedB[byteIdx >> 2] : 0u;
                uint p1 = (kValid && tileN + b_col + 4u < pc.N) ? packedB[(byteIdx + 4u) >> 2] : 0u;
                int sp0 = (int)p0;
                int sp1 = (int)p1;
                tileB[b_row][b_col + 0u] = (signed char)((sp0 << 24) >> 24);
                tileB[b_row][b_col + 1u] = (signed char)((sp0 << 16) >> 24);
                tileB[b_row][b_col + 2u] = (signed char)((sp0 <<  8) >> 24);
                tileB[b_row][b_col + 3u] = (signed char)(sp0 >> 24);
                tileB[b_row][b_col + 4u] = (signed char)((sp1 << 24) >> 24);
                tileB[b_row][b_col + 5u] = (signed char)((sp1 << 16) >> 24);
                tileB[b_row][b_col + 6u] = (signed char)((sp1 <<  8) >> 24);
                tileB[b_row][b_col + 7u] = (signed char)(sp1 >> 24);
            }

            // STEP 2: Load B scales
            if (tid < 64u) {
                uint sidx = numBlocks * pc.scaleStride + tileN + tid;
                b_scales[tid] = (tileN + tid < pc.N) ? float(scalesB[sidx >> 2][sidx & 3]) : 0.0f;
            }

            // STEP 3: Load A float, quantize per-row to int8
            {
                bool rowValid = (tileM + a_row < pc.M);
                float a0 = (rowValid && k_base + a_k_local + 0u < pc.K) ? (float)cut_loadA_fast(dataA, pc, tileM + a_row, k_base + a_k_local + 0u) : 0.0f;
                float a1 = (rowValid && k_base + a_k_local + 1u < pc.K) ? (float)cut_loadA_fast(dataA, pc, tileM + a_row, k_base + a_k_local + 1u) : 0.0f;
                float a2 = (rowValid && k_base + a_k_local + 2u < pc.K) ? (float)cut_loadA_fast(dataA, pc, tileM + a_row, k_base + a_k_local + 2u) : 0.0f;
                float a3 = (rowValid && k_base + a_k_local + 3u < pc.K) ? (float)cut_loadA_fast(dataA, pc, tileM + a_row, k_base + a_k_local + 3u) : 0.0f;

                float lm = fmaxf(fmaxf(fabsf(a0), fabsf(a1)), fmaxf(fabsf(a2), fabsf(a3)));
                lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, 4u));
                lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, 2u));
                lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, 1u));

                float a_scale_inv = (lm > 0.0f) ? (127.0f / lm) : 0.0f;
                if ((tid & 7u) == 0u) a_scales[a_row] = lm / 127.0f;

                tileA[a_row][a_k_local + 0u] = (signed char)clamp((int)round(a0 * a_scale_inv), -127, 127);
                tileA[a_row][a_k_local + 1u] = (signed char)clamp((int)round(a1 * a_scale_inv), -127, 127);
                tileA[a_row][a_k_local + 2u] = (signed char)clamp((int)round(a2 * a_scale_inv), -127, 127);
                tileA[a_row][a_k_local + 3u] = (signed char)clamp((int)round(a3 * a_scale_inv), -127, 127);
            }

            __syncthreads();

            // STEP 4: Compute integer dot products for this K-tile
            {
                int acc_int[2][4];
                for (uint m = 0; m < 2u; m++)
                    for (uint n = 0; n < 4u; n++)
                        acc_int[m][n] = 0;

                for (uint k = 0u; k < 32u; k += 4u) {
                    uint a_p0 = cut_pack_i8x4(tileA[localRow][k],     tileA[localRow][k+1u],     tileA[localRow][k+2u],     tileA[localRow][k+3u]);
                    uint a_p1 = cut_pack_i8x4(tileA[localRow+16u][k], tileA[localRow+16u][k+1u], tileA[localRow+16u][k+2u], tileA[localRow+16u][k+3u]);
                    uint b_p0 = cut_pack_i8x4(tileB[k][localCol],     tileB[k+1u][localCol],     tileB[k+2u][localCol],     tileB[k+3u][localCol]);
                    uint b_p1 = cut_pack_i8x4(tileB[k][localCol+16u], tileB[k+1u][localCol+16u], tileB[k+2u][localCol+16u], tileB[k+3u][localCol+16u]);
                    uint b_p2 = cut_pack_i8x4(tileB[k][localCol+32u], tileB[k+1u][localCol+32u], tileB[k+2u][localCol+32u], tileB[k+3u][localCol+32u]);
                    uint b_p3 = cut_pack_i8x4(tileB[k][localCol+48u], tileB[k+1u][localCol+48u], tileB[k+2u][localCol+48u], tileB[k+3u][localCol+48u]);

                    acc_int[0][0] = __dp4a((int)a_p0, (int)b_p0, acc_int[0][0]);
                    acc_int[0][1] = __dp4a((int)a_p0, (int)b_p1, acc_int[0][1]);
                    acc_int[0][2] = __dp4a((int)a_p0, (int)b_p2, acc_int[0][2]);
                    acc_int[0][3] = __dp4a((int)a_p0, (int)b_p3, acc_int[0][3]);
                    acc_int[1][0] = __dp4a((int)a_p1, (int)b_p0, acc_int[1][0]);
                    acc_int[1][1] = __dp4a((int)a_p1, (int)b_p1, acc_int[1][1]);
                    acc_int[1][2] = __dp4a((int)a_p1, (int)b_p2, acc_int[1][2]);
                    acc_int[1][3] = __dp4a((int)a_p1, (int)b_p3, acc_int[1][3]);
                }

                for (uint m = 0; m < 2u; m++) {
                    float as = a_scales[localRow + m * 16u];
                    for (uint n = 0; n < 4u; n++) {
                        float bs = b_scales[localCol + n * 16u];
                        acc[m][n] += (float)acc_int[m][n] * as * bs;
                    }
                }
            }

            __syncthreads();
        }
    }

    // Write outputs
    for (uint m = 0; m < 2u; m++) {
        uint outRow = tileM + localRow + m * 16u;
        if (outRow >= pc.M) continue;
        for (uint n = 0; n < 4u; n++) {
            uint outCol = tileN + localCol + n * 16u;
            if (outCol >= pc.N) continue;
            cut_writeOutput(dataC, dataD, pc, outRow, outCol, acc[m][n]);
        }
    }
}
