// Native CUDA counterpart of Conv2DImplicitGemmS3.shader — keep semantics in
// lockstep.
//
// 3x3 stride-1 convolution, spatially tiled. Same register-tiled structure as
// Conv2DImplicitGemm.cu, with one change that is worth a whole kernel:
//
// The implicit-GEMM kernel stages the im2col matrix, so an input pixel arrives
// in shared memory nine times (once per tap) and is read out of shared once per
// tap per output it feeds. That kernel is exactly balanced between the FMA
// pipes and shared-memory bandwidth — 512 FFMA and 32 LDS.128 per k-tile per
// warp both work out to ~2048 SM-cycles per k-tile round on a 3090 — so it runs
// at about half the fp32 peak no matter how well the arithmetic is scheduled.
//
// Here the block stages an (S3_TH + 2) x (S3_TW + 2) patch of *pixels* instead,
// and each thread pulls its 6 columns into registers once and slides the 3x3
// window over them. Per input channel a thread reads 36 input scalars and 72
// weights for 576 FMAs (0.19 scalars/FMA) where the GEMM reads 0.25 — enough to
// take shared memory off the critical path and leave the FMA pipes as the only
// limit.
//
// Restricted to kH = kW = 3, strideH = strideW = 1 (any padding); the sliding
// window only overlaps at unit stride, and Conv2DOpNode::defaultVariant sends
// everything else to the general implicit-GEMM variant.
//
// Dispatch: x = tiles along W_out, y = batch * tiles along H_out, z = tiles
// along C_out.

#include "Conv2DCommon.cuh"

#ifndef S3_TH
#define S3_TH 8 // output rows per block
#endif
#ifndef S3_TW
#define S3_TW 16 // output columns per block
#endif
#ifndef S3_BN
#define S3_BN 128 // output channels per block
#endif
#ifndef S3_BK
#define S3_BK 4 // input channels per k-tile
#endif
#ifndef S3_TMH
#define S3_TMH 2 // output rows per thread
#endif
#ifndef S3_TMW
#define S3_TMW 4 // output columns per thread
#endif
#ifndef S3_TN
#define S3_TN 8 // output channels per thread
#endif

#define S3_PGH (S3_TH / S3_TMH) // position groups down
#define S3_PGW (S3_TW / S3_TMW) // position groups across
#define S3_CG (S3_BN / S3_TN)   // channel groups
#define S3_THREADS (S3_PGH * S3_PGW * S3_CG)

// Staged input patch. The row stride is padded to 24 rather than the 18 columns
// actually used: rows must start 16-byte aligned for the compute loop's
// LDS.128, and at 24 the eight threads of an LDS phase (two position rows, four
// position columns) land on 8 * 4 distinct banks instead of colliding in pairs.
#define S3_SH (S3_TH + 2)
#define S3_SW 24

// Weight tile, padded likewise: the loader walks taps fastest (which is how
// weights are laid out in memory), and at an unpadded S3_BN every tap of one
// channel would hit the same bank on the way in.
#define S3_BNP (S3_BN + 4)

extern "C" __global__ __launch_bounds__(S3_THREADS) void cut_main(
    const cut_conv2d_t* __restrict__ input_data,
    const cut_conv2d_t* __restrict__ weight_data,
    cut_conv2d_t* __restrict__ output_data,
    PushConstants pc) {
    __shared__ __align__(16) cut_conv2d_t As[S3_BK][S3_SH][S3_SW];
    __shared__ __align__(16) cut_conv2d_t Bs[S3_BK][9][S3_BNP];

    const uint H_out = pc.H_in + 2 * pc.padH - 2;
    const uint W_out = pc.W_in + 2 * pc.padW - 2;
    const uint inAlignedW = (pc.W_in + 3) & ~3u;
    const uint outAlignedW = (W_out + 3) & ~3u;
    const uint weightAlignedKW = (pc.kW + 3) & ~3u;

    const uint tid = threadIdx.x;
    const uint hTiles = (H_out + S3_TH - 1) / S3_TH;
    const uint nb = blockIdx.y / hTiles;
    const uint h0 = (blockIdx.y - nb * hTiles) * S3_TH;
    const uint w0 = blockIdx.x * S3_TW;
    const uint nBase = blockIdx.z * S3_BN;

    // Top-left input pixel staged by this block.
    const int hBase = int(h0) - int(pc.padH);
    const int wBase = int(w0) - int(pc.padW);

    const uint inChanStride = pc.H_in * inAlignedW;
    const uint inBatchBase = nb * pc.C_in * inChanStride;
    const uint wChanStride = pc.kH * weightAlignedKW;

    // pg indexes the output patch, cg the output channels.
    const uint pg = tid % (S3_PGH * S3_PGW);
    const uint cg = tid / (S3_PGH * S3_PGW);
    const uint pgh = pg / S3_PGW;
    const uint pgw = pg - pgh * S3_PGW;

    cut_conv2d_t acc[S3_TMH][S3_TMW][S3_TN];
#pragma unroll
    for (int i = 0; i < S3_TMH; ++i)
#pragma unroll
        for (int q = 0; q < S3_TMW; ++q)
#pragma unroll
            for (int j = 0; j < S3_TN; ++j)
                acc[i][q][j] = (cut_conv2d_t)(0);

    for (uint c0 = 0; c0 < pc.C_in; c0 += S3_BK) {
        // Guards the previous iteration's reads of the tiles being overwritten.
        __syncthreads();

        // --- stage the input patch: S3_BK channels x S3_SH x (S3_TW + 2) ----
        // Flat walk with compile-time constant divisors, so consecutive threads
        // take consecutive columns of one input row.
        for (uint idx = tid; idx < S3_BK * S3_SH * (S3_TW + 2);
             idx += S3_THREADS) {
            const uint c = idx / (S3_SH * (S3_TW + 2));
            const uint r2 = idx - c * (S3_SH * (S3_TW + 2));
            const uint row = r2 / (S3_TW + 2);
            const uint col = r2 - row * (S3_TW + 2);
            const int ih = hBase + int(row);
            const int iw = wBase + int(col);
            cut_conv2d_t v = (cut_conv2d_t)(0);
            if (c0 + c < pc.C_in && ih >= 0 && ih < int(pc.H_in) && iw >= 0 &&
                iw < int(pc.W_in))
                v = input_data[inBatchBase + (c0 + c) * inChanStride
                               + uint(ih) * inAlignedW + uint(iw)];
            As[c][row][col] = v;
        }

        // --- stage the weights: S3_BK channels x 9 taps x S3_BN channels ----
        for (uint idx = tid; idx < S3_BK * 9 * S3_BN; idx += S3_THREADS) {
            const uint tap = idx % 9;
            const uint t2 = idx / 9;
            const uint c = t2 % S3_BK;
            const uint n = t2 / S3_BK;
            const uint co = nBase + n;
            cut_conv2d_t v = (cut_conv2d_t)(0);
            if (co < pc.C_out && c0 + c < pc.C_in)
                v = weight_data[co * pc.C_in * wChanStride
                                + (c0 + c) * wChanStride
                                + (tap / 3) * weightAlignedKW + (tap % 3)];
            Bs[c][tap][n] = v;
        }
        __syncthreads();

#pragma unroll
        for (int c = 0; c < S3_BK; ++c) {
            // The thread's own slice of the patch: S3_TMH + 2 rows by
            // S3_TMW + 2 columns, read once and reused across all nine taps.
            cut_conv2d_t p[S3_TMH + 2][S3_TMW + 2];
#pragma unroll
            for (int r = 0; r < S3_TMH + 2; ++r)
#pragma unroll
                for (int q = 0; q < S3_TMW + 2; ++q)
                    p[r][q] = As[c][pgh * S3_TMH + uint(r)][pgw * S3_TMW + uint(q)];

#pragma unroll
            for (int kh = 0; kh < 3; ++kh)
#pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    cut_conv2d_t bf[S3_TN];
#pragma unroll
                    for (int j = 0; j < S3_TN / 2; ++j) {
                        bf[j] = Bs[c][kh * 3 + kw][cg * (S3_TN / 2) + uint(j)];
                        bf[j + S3_TN / 2] =
                            Bs[c][kh * 3 + kw]
                              [S3_BN / 2 + cg * (S3_TN / 2) + uint(j)];
                    }
#pragma unroll
                    for (int i = 0; i < S3_TMH; ++i)
#pragma unroll
                        for (int q = 0; q < S3_TMW; ++q)
#pragma unroll
                            for (int j = 0; j < S3_TN; ++j)
                                acc[i][q][j] += p[i + kh][q + kw] * bf[j];
                }
        }
    }

    const uint outChanStride = H_out * outAlignedW;
#pragma unroll
    for (int i = 0; i < S3_TMH; ++i) {
        const uint ho = h0 + pgh * S3_TMH + uint(i);
        if (ho >= H_out)
            continue;
#pragma unroll
        for (int q = 0; q < S3_TMW; ++q) {
            const uint wo = w0 + pgw * S3_TMW + uint(q);
            if (wo >= W_out)
                continue;
            const uint rowBase =
                nb * pc.C_out * outChanStride + ho * outAlignedW + wo;
#pragma unroll
            for (int j = 0; j < S3_TN; ++j) {
                const uint co =
                    nBase + ((j < S3_TN / 2)
                                 ? (cg * (S3_TN / 2) + uint(j))
                                 : (S3_BN / 2 + cg * (S3_TN / 2) + uint(j - S3_TN / 2)));
                if (co < pc.C_out)
                    output_data[rowBase + co * outChanStride] = acc[i][q][j];
            }
        }
    }
}
