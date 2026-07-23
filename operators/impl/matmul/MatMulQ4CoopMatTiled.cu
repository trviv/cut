// Native CUDA counterpart of MatMulQ4CoopMatTiled.comp: 4-bit weights dequantized
// to fp16 in shared memory, then 16x16x16 fp16 tensor-core MMA (fp32 accumulate).
// 128-thread (4-warp) workgroup, 2x2 grid of 16x16 tiles = 32x32 output tile.
// Tensor-core MMA via inline-PTX helpers (mma.h is unusable under NVRTC).
#include "MatMulCoopMatCommon.cuh"

extern "C" __global__ void cut_main(const float* __restrict__ dataA,
                                    const unsigned int* __restrict__ packedB,
                                    const half* __restrict__ scalesB,
                                    const float* __restrict__ dataD,
                                    float* __restrict__ dataC,
                                    Q4CoopMatPush pc) {
    __shared__ __align__(16) half tileA_sh[32 * 16];   // 32 rows x 16 K
    __shared__ half tileB_sh[16 * 32];                 // 16 K x 32 cols
    __shared__ float tileC_sh[32 * 32];                // 32 x 32 output

    uint tid = threadIdx.x;      // 0..127
    uint sgId = tid >> 5;        // warp id 0..3
    int lane = (int)(tid & 31u);

    uint subTileRow = sgId / 2u;   // 0 or 1
    uint subTileCol = sgId % 2u;   // 0 or 1

    uint tileRowStart = blockIdx.y * 32u;
    uint tileColStart = blockIdx.x * 32u;

    uint tileRow = tileRowStart + subTileRow * 16u;
    uint tileCol = tileColStart + subTileCol * 16u;

    bool validTile = (tileRow < pc.M && tileCol < pc.N);

    Acc16 acc;
    zeroAcc16(acc);

    for (uint k = 0; k < pc.K; k += 16u) {
        // Load A tile (32x16): float32 -> fp16. 512 elements, 128 threads -> 4 each.
        for (uint i = tid; i < 512u; i += 128u) {
            uint row = i >> 4;   // 0..31
            uint col = i & 15u;  // 0..15
            uint gRow = tileRowStart + row;
            float val = (gRow < pc.M) ? dataA[gRow * pc.strideA + k + col] : 0.0f;
            tileA_sh[i] = (half)val;
        }

        // Load B tile (16x32): packed 4-bit -> dequant -> fp16. 128 threads each
        // load one uint32 (2 bytes, 4 nibbles) + 4 scales -> 4 fp16.
        {
            uint kRow = tid >> 3;      // 0..15
            uint nGroup = tid & 7u;    // 0..7
            uint nCol = nGroup << 2;   // 0,4,...,28

            uint gK = k + kRow;
            uint gN = tileColStart + nCol;

            uint byteIdx = gK * pc.strideBNpacked + (gN >> 1u);
            unsigned int packed = packedB[byteIdx >> 2];
            uint shift = (byteIdx & 3u) * 8u;
            uint twoBytes = (packed >> shift) & 0xFFFFu;

            int b0 = int((twoBytes >>  0) & 0xFu) - 8;
            int b1 = int((twoBytes >>  4) & 0xFu) - 8;
            int b2 = int((twoBytes >>  8) & 0xFu) - 8;
            int b3 = int((twoBytes >> 12) & 0xFu) - 8;

            uint scaleBase = (gK >> 5) * pc.scaleStride + gN;
            half s0 = scalesB[scaleBase];
            half s1 = scalesB[scaleBase + 1u];
            half s2 = scalesB[scaleBase + 2u];
            half s3 = scalesB[scaleBase + 3u];

            half m0 = 0.0f;
            half m1 = 0.0f;
            half m2 = 0.0f;
            half m3 = 0.0f;
            if (COOPMAT_AFFINE == 1u) {
                uint blocksK = (pc.K + 31u) >> 5u;
                m0 = scalesB[blocksK * pc.scaleStride + scaleBase];
                m1 = scalesB[blocksK * pc.scaleStride + scaleBase + 1u];
                m2 = scalesB[blocksK * pc.scaleStride + scaleBase + 2u];
                m3 = scalesB[blocksK * pc.scaleStride + scaleBase + 3u];
            }

            uint bBase = kRow * 32u + nCol;
            tileB_sh[bBase + 0] = (half)((float)b0 * (float)s0 + (float)m0);
            tileB_sh[bBase + 1] = (half)((float)b1 * (float)s1 + (float)m1);
            tileB_sh[bBase + 2] = (half)((float)b2 * (float)s2 + (float)m2);
            tileB_sh[bBase + 3] = (half)((float)b3 * (float)s3 + (float)m3);
        }

        __syncthreads();

        if (validTile) {
            FragA16 matA;
            cutLoadA16(matA, tileA_sh + subTileRow * 16u * 16u, 16, lane);
            FragB16 matB;
            cutLoadB16(matB, tileB_sh + subTileCol * 16u, 32, lane);
            cutMma16(acc, matA, matB);
        }

        __syncthreads();
    }

    // Store 16x16 accumulator into the 32x32 shared output tile.
    if (validTile) {
        uint cOffset = subTileRow * 16u * 32u + subTileCol * 16u;
        cutStoreC16(tileC_sh + cOffset, 32, acc, lane);
    }

    __syncthreads();

    // Copy valid results from shared memory to global output (handles non-aligned M/N),
    // applying the fused unary/binary epilogue so fusion does not force a
    // fallback off this tensor-core variant.
    for (uint i = tid; i < 32u * 32u; i += 128u) {
        uint row = i / 32u;
        uint col = i % 32u;
        uint gRow = tileRowStart + row;
        uint gCol = tileColStart + col;
        if (gRow < pc.M && gCol < pc.N) {
            uint idx = gRow * pc.strideC + gCol;
            float d = (COOPMAT_FUSION_TYPE == 2u) ? dataD[idx] : 0.0f;
            dataC[idx] = cutApplyFusion(tileC_sh[i], d);
        }
    }
}
