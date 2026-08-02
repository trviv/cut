#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Implicit-GEMM Conv2D. See Conv2DImplicitGemm.cu for the full derivation; in
// short, the forward convolution is run as C[m,n] = sum_k A[m,k]*B[k,n] with
//   m = (batch, h_out, w_out), n = c_out, k = (c_in, kh, kw)
// and A gathered from the input on the fly rather than materialised as im2col.
//
// This is the same algorithm as the .cu, single-buffered: the CUDA path double
// buffers the shared tiles, which is worth a barrier per k-tile there and is
// not expressible as cheaply here. Dispatch: x = tiles along M, y = tiles along
// C_out.

#define GEMM_BM %GEMM_BM%
#define GEMM_BN %GEMM_BN%
#define GEMM_BK %GEMM_BK%
#define GEMM_TM %GEMM_TM%
#define GEMM_TN %GEMM_TN%

#define GEMM_THREADS ((GEMM_BM / GEMM_TM) * (GEMM_BN / GEMM_TN))
#define GEMM_A_KPT (GEMM_BK / (GEMM_THREADS / GEMM_BM))
#define GEMM_B_TPN (GEMM_THREADS / GEMM_BN)
#define GEMM_B_KPT (GEMM_BK / GEMM_B_TPN)
#define GEMM_TMH (GEMM_TM / 2)
#define GEMM_TNH (GEMM_TN / 2)

#include "Conv2DCommon.shaderh"

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> input_data;

[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> weight_data;

[[vk::binding(2, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> output_data;

groupshared %SCALAR_DTYPE_INPUT% As[GEMM_BK][GEMM_BM];
groupshared %SCALAR_DTYPE_INPUT% Bs[GEMM_BK][GEMM_BN];

[numthreads(GEMM_THREADS, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint H_out = (pc.H_in + 2 * pc.padH - pc.kH) / pc.strideH + 1;
    uint W_out = (pc.W_in + 2 * pc.padW - pc.kW) / pc.strideW + 1;
    uint inAlignedW = (pc.W_in + 3) & ~3u;
    uint outAlignedW = (W_out + 3) & ~3u;
    uint weightAlignedKW = (pc.kW + 3) & ~3u;

    uint kHW = pc.kH * pc.kW;
    uint K = pc.C_in * kHW;
    uint HW = H_out * W_out;
    uint M = pc.batchSize * HW;

    uint tid = GTid.x;
    uint mBase = Gid.x * GEMM_BM;
    uint nBase = Gid.y * GEMM_BN;

    // A loader: one output position, GEMM_A_KPT consecutive k.
    uint aM = tid % GEMM_BM;
    uint aKOff = (tid / GEMM_BM) * GEMM_A_KPT;
    uint aRow = mBase + aM;
    bool aValid = aRow < M;
    uint aChanStride = pc.H_in * inAlignedW;
    uint aBatchBase;
    int aH0, aW0;
    {
        uint m = aValid ? aRow : 0u;
        uint nb = m / HW;
        uint rem = m - nb * HW;
        uint ho = rem / W_out;
        uint wo = rem - ho * W_out;
        aBatchBase = nb * pc.C_in * aChanStride;
        aH0 = int(ho * pc.strideH) - int(pc.padH);
        aW0 = int(wo * pc.strideW) - int(pc.padW);
    }

    // B loader: one output channel, GEMM_B_KPT consecutive k.
    uint bN = tid / GEMM_B_TPN;
    uint bKOff = (tid % GEMM_B_TPN) * GEMM_B_KPT;
    uint bCol = nBase + bN;
    bool bValid = bCol < pc.C_out;
    uint bChanStride = pc.kH * weightAlignedKW;
    uint bColBase = bValid ? bCol * pc.C_in * bChanStride : 0u;

    // k = (c_in, kh, kw) carried incrementally: decode once, then advance by
    // GEMM_BK per tile with a precomputed delta and two carries.
    uint dci, dkh, dkw;
    {
        uint c = GEMM_BK / kHW;
        uint r = GEMM_BK - c * kHW;
        uint h = r / pc.kW;
        dci = c;
        dkh = h;
        dkw = r - h * pc.kW;
    }
    uint aci, akh, akw, bci, bkh, bkw;
    {
        uint c = aKOff / kHW;
        uint r = aKOff - c * kHW;
        uint h = r / pc.kW;
        aci = c; akh = h; akw = r - h * pc.kW;
        c = bKOff / kHW; r = bKOff - c * kHW; h = r / pc.kW;
        bci = c; bkh = h; bkw = r - h * pc.kW;
    }

    %SCALAR_DTYPE_INPUT% acc[GEMM_TM][GEMM_TN];
    [unroll] for (int ai = 0; ai < GEMM_TM; ++ai)
        [unroll] for (int aj = 0; aj < GEMM_TN; ++aj)
            acc[ai][aj] = (%SCALAR_DTYPE_INPUT%)(0);

    uint tm = tid % (GEMM_BM / GEMM_TM);
    uint tn = tid / (GEMM_BM / GEMM_TM);
    uint aFrag0 = tm * GEMM_TMH;
    uint bFrag0 = tn * GEMM_TNH;

    for (uint k0 = 0; k0 < K; k0 += GEMM_BK) {
        // Stage this k-tile: A transposed (k-major) so the compute loop reads
        // GEMM_TMH contiguous scalars per thread.
        {
            uint ci = aci, kh = akh, kw = akw;
            [unroll] for (int i = 0; i < GEMM_A_KPT; ++i) {
                %SCALAR_DTYPE_INPUT% v = (%SCALAR_DTYPE_INPUT%)(0);
                if (aValid && k0 + aKOff + uint(i) < K) {
                    int ih = aH0 + int(kh);
                    int iw = aW0 + int(kw);
                    if (ih >= 0 && ih < int(pc.H_in) && iw >= 0 && iw < int(pc.W_in))
                        v = input_data[aBatchBase + ci * aChanStride
                                       + uint(ih) * inAlignedW + uint(iw)];
                }
                As[aKOff + uint(i)][aM] = v;
                kw++;
                if (kw == pc.kW) { kw = 0; kh++; if (kh == pc.kH) { kh = 0; ci++; } }
            }

            ci = bci; kh = bkh; kw = bkw;
            [unroll] for (int i = 0; i < GEMM_B_KPT; ++i) {
                %SCALAR_DTYPE_INPUT% v = (%SCALAR_DTYPE_INPUT%)(0);
                if (bValid && k0 + bKOff + uint(i) < K)
                    v = weight_data[bColBase + ci * bChanStride
                                    + kh * weightAlignedKW + kw];
                Bs[bKOff + uint(i)][bN] = v;
                kw++;
                if (kw == pc.kW) { kw = 0; kh++; if (kh == pc.kH) { kh = 0; ci++; } }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (int kk = 0; kk < GEMM_BK; ++kk) {
            %SCALAR_DTYPE_INPUT% af[GEMM_TM];
            %SCALAR_DTYPE_INPUT% bf[GEMM_TN];
            [unroll] for (int i = 0; i < GEMM_TMH; ++i) {
                af[i] = As[kk][aFrag0 + uint(i)];
                af[i + GEMM_TMH] = As[kk][GEMM_BM / 2 + aFrag0 + uint(i)];
            }
            [unroll] for (int j = 0; j < GEMM_TNH; ++j) {
                bf[j] = Bs[kk][bFrag0 + uint(j)];
                bf[j + GEMM_TNH] = Bs[kk][GEMM_BN / 2 + bFrag0 + uint(j)];
            }
            [unroll] for (int i2 = 0; i2 < GEMM_TM; ++i2)
                [unroll] for (int j2 = 0; j2 < GEMM_TN; ++j2)
                    acc[i2][j2] += af[i2] * bf[j2];
        }
        GroupMemoryBarrierWithGroupSync();

        akw += dkw;
        uint ac1 = (akw >= pc.kW) ? 1u : 0u;
        akw -= ac1 * pc.kW;
        akh += dkh + ac1;
        uint ac2 = (akh >= pc.kH) ? 1u : 0u;
        akh -= ac2 * pc.kH;
        aci += dci + ac2;

        bkw += dkw;
        uint bc1 = (bkw >= pc.kW) ? 1u : 0u;
        bkw -= bc1 * pc.kW;
        bkh += dkh + bc1;
        uint bc2 = (bkh >= pc.kH) ? 1u : 0u;
        bkh -= bc2 * pc.kH;
        bci += dci + bc2;
    }

    uint outChanStride = H_out * outAlignedW;
    [unroll] for (int i = 0; i < GEMM_TM; ++i) {
        uint mo = mBase + ((i < GEMM_TMH)
                               ? (aFrag0 + uint(i))
                               : (GEMM_BM / 2 + aFrag0 + uint(i - GEMM_TMH)));
        if (mo < M) {
            uint nb = mo / HW;
            uint rem = mo - nb * HW;
            uint ho = rem / W_out;
            uint wo = rem - ho * W_out;
            uint rowBase = nb * pc.C_out * outChanStride + ho * outAlignedW + wo;
            [unroll] for (int j = 0; j < GEMM_TN; ++j) {
                uint co = nBase + ((j < GEMM_TNH)
                                       ? (bFrag0 + uint(j))
                                       : (GEMM_BN / 2 + bFrag0 + uint(j - GEMM_TNH)));
                if (co < pc.C_out)
                    output_data[rowBase + co * outChanStride] = acc[i][j];
            }
        }
    }
}
