#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// 3x3 stride-1 convolution, spatially tiled. See Conv2DImplicitGemmS3.cu for
// the derivation; in short, the block stages an (S3_TH + 2) x (S3_TW + 2) patch
// of input pixels rather than the im2col matrix, and each thread slides the 3x3
// window over six columns held in registers. That is ~1.5x less shared-memory
// traffic per FMA than the general implicit-GEMM variant, which is what the
// direct kernel is limited by.
//
// Requires kH = kW = 3 and strideH = strideW = 1; padding is arbitrary.
// Dispatch: x = tiles along W_out, y = batch * tiles along H_out, z = tiles
// along C_out.

#define S3_TH %S3_TH%
#define S3_TW %S3_TW%
#define S3_BN %S3_BN%
#define S3_BK %S3_BK%
#define S3_TMH %S3_TMH%
#define S3_TMW %S3_TMW%
#define S3_TN %S3_TN%

#define S3_PGH (S3_TH / S3_TMH)
#define S3_PGW (S3_TW / S3_TMW)
#define S3_CG (S3_BN / S3_TN)
#define S3_THREADS (S3_PGH * S3_PGW * S3_CG)

#define S3_SH (S3_TH + 2)
#define S3_SW 24
#define S3_BNP (S3_BN + 4)

#include "Conv2DCommon.shaderh"

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> input_data;

[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> weight_data;

[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> output_data;

groupshared %SCALAR_DTYPE_INPUT% As[S3_BK][S3_SH][S3_SW];
groupshared %SCALAR_DTYPE_INPUT% Bs[S3_BK][9][S3_BNP];

[numthreads(S3_THREADS, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint H_out = pc.H_in + 2 * pc.padH - 2;
    uint W_out = pc.W_in + 2 * pc.padW - 2;
    uint inAlignedW = (pc.W_in + 3) & ~3u;
    uint outAlignedW = (W_out + 3) & ~3u;
    uint weightAlignedKW = (pc.kW + 3) & ~3u;

    uint tid = GTid.x;
    uint hTiles = (H_out + S3_TH - 1) / S3_TH;
    uint nb = Gid.y / hTiles;
    uint h0 = (Gid.y - nb * hTiles) * S3_TH;
    uint w0 = Gid.x * S3_TW;
    uint nBase = Gid.z * S3_BN;

    int hBase = int(h0) - int(pc.padH);
    int wBase = int(w0) - int(pc.padW);

    uint inChanStride = pc.H_in * inAlignedW;
    uint inBatchBase = nb * pc.C_in * inChanStride;
    uint wChanStride = pc.kH * weightAlignedKW;

    uint pg = tid % (S3_PGH * S3_PGW);
    uint cg = tid / (S3_PGH * S3_PGW);
    uint pgh = pg / S3_PGW;
    uint pgw = pg - pgh * S3_PGW;

    %SCALAR_DTYPE_INPUT% acc[S3_TMH][S3_TMW][S3_TN];
    [unroll] for (int ai = 0; ai < S3_TMH; ++ai)
        [unroll] for (int aq = 0; aq < S3_TMW; ++aq)
            [unroll] for (int aj = 0; aj < S3_TN; ++aj)
                acc[ai][aq][aj] = (%SCALAR_DTYPE_INPUT%)(0);

    for (uint c0 = 0; c0 < pc.C_in; c0 += S3_BK) {
        GroupMemoryBarrierWithGroupSync();

        for (uint idx = tid; idx < S3_BK * S3_SH * (S3_TW + 2); idx += S3_THREADS) {
            uint c = idx / (S3_SH * (S3_TW + 2));
            uint r2 = idx - c * (S3_SH * (S3_TW + 2));
            uint row = r2 / (S3_TW + 2);
            uint col = r2 - row * (S3_TW + 2);
            int ih = hBase + int(row);
            int iw = wBase + int(col);
            %SCALAR_DTYPE_INPUT% v = (%SCALAR_DTYPE_INPUT%)(0);
            if (c0 + c < pc.C_in && ih >= 0 && ih < int(pc.H_in) && iw >= 0 && iw < int(pc.W_in))
                v = input_data[inBatchBase + (c0 + c) * inChanStride
                               + uint(ih) * inAlignedW + uint(iw)];
            As[c][row][col] = v;
        }

        for (uint widx = tid; widx < S3_BK * 9 * S3_BN; widx += S3_THREADS) {
            uint tap = widx % 9;
            uint t2 = widx / 9;
            uint c = t2 % S3_BK;
            uint n = t2 / S3_BK;
            uint co = nBase + n;
            %SCALAR_DTYPE_INPUT% v = (%SCALAR_DTYPE_INPUT%)(0);
            if (co < pc.C_out && c0 + c < pc.C_in)
                v = weight_data[co * pc.C_in * wChanStride + (c0 + c) * wChanStride
                                + (tap / 3) * weightAlignedKW + (tap % 3)];
            Bs[c][tap][n] = v;
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (int c2 = 0; c2 < S3_BK; ++c2) {
            %SCALAR_DTYPE_INPUT% p[S3_TMH + 2][S3_TMW + 2];
            [unroll] for (int r = 0; r < S3_TMH + 2; ++r)
                [unroll] for (int q = 0; q < S3_TMW + 2; ++q)
                    p[r][q] = As[c2][pgh * S3_TMH + uint(r)][pgw * S3_TMW + uint(q)];

            [unroll] for (int kh = 0; kh < 3; ++kh)
                [unroll] for (int kw = 0; kw < 3; ++kw) {
                    %SCALAR_DTYPE_INPUT% bf[S3_TN];
                    [unroll] for (int j = 0; j < S3_TN / 2; ++j) {
                        bf[j] = Bs[c2][kh * 3 + kw][cg * (S3_TN / 2) + uint(j)];
                        bf[j + S3_TN / 2] = Bs[c2][kh * 3 + kw][S3_BN / 2 + cg * (S3_TN / 2) + uint(j)];
                    }
                    [unroll] for (int i2 = 0; i2 < S3_TMH; ++i2)
                        [unroll] for (int q2 = 0; q2 < S3_TMW; ++q2)
                            [unroll] for (int j2 = 0; j2 < S3_TN; ++j2)
                                acc[i2][q2][j2] += p[i2 + kh][q2 + kw] * bf[j2];
                }
        }
    }

    uint outChanStride = H_out * outAlignedW;
    [unroll] for (int i = 0; i < S3_TMH; ++i) {
        uint ho = h0 + pgh * S3_TMH + uint(i);
        if (ho < H_out) {
            [unroll] for (int q = 0; q < S3_TMW; ++q) {
                uint wo = w0 + pgw * S3_TMW + uint(q);
                if (wo < W_out) {
                    uint rowBase = nb * pc.C_out * outChanStride + ho * outAlignedW + wo;
                    [unroll] for (int j = 0; j < S3_TN; ++j) {
                        uint co = nBase + ((j < S3_TN / 2)
                                               ? (cg * (S3_TN / 2) + uint(j))
                                               : (S3_BN / 2 + cg * (S3_TN / 2) + uint(j - S3_TN / 2)));
                        if (co < pc.C_out)
                            output_data[rowBase + co * outChanStride] = acc[i][q][j];
                    }
                }
            }
        }
    }
}
