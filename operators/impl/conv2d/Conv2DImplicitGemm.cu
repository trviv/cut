// Native CUDA counterpart of Conv2DImplicitGemm.shader — keep semantics in
// lockstep (the HLSL is the same algorithm without the double buffering).
//
// Forward convolution written as the GEMM it actually is, with the im2col
// matrix gathered on the fly instead of materialised:
//
//     C[m, n] = sum_k A[m, k] * B[k, n]
//       m = (batch, h_out, w_out)   -> M = N * H_out * W_out
//       n = c_out                   -> N_gemm = C_out
//       k = (c_in, kh, kw)          -> K = C_in * kH * kW
//
// A[m, k] is one input pixel and B[k, n] one weight, so no scratch buffer and
// no extra pass are needed. Both operands are staged through shared memory in
// GEMM_BM x GEMM_BK / GEMM_BK x GEMM_BN tiles and each thread keeps a
// GEMM_TM x GEMM_TN register tile, which is where the arithmetic intensity the
// direct kernels lack comes from: the naive variant re-reads every input pixel
// and every weight once per output element.
//
// Block is 1-D, GEMM_THREADS threads. Dispatch (see Conv2DOpNode::dispatchSize)
// is x = tiles along M, y = tiles along C_out — M on x because it is the axis
// that grows without bound and gridDim.y caps at 65535.

#include "Conv2DCommon.cuh"

#ifndef GEMM_BM
#define GEMM_BM 128
#endif
#ifndef GEMM_BN
#define GEMM_BN 128
#endif
#ifndef GEMM_BK
#define GEMM_BK 8
#endif
#ifndef GEMM_TM
#define GEMM_TM 8
#endif
#ifndef GEMM_TN
#define GEMM_TN 8
#endif

#define GEMM_THREADS ((GEMM_BM / GEMM_TM) * (GEMM_BN / GEMM_TN))

// A tile: GEMM_THREADS/GEMM_BM threads cover the k axis, so consecutive threads
// walk consecutive m — i.e. consecutive w_out, which is contiguous in NCHW.
#define GEMM_A_KPT (GEMM_BK / (GEMM_THREADS / GEMM_BM))
// B tile: the other way round. Weights are contiguous along k, so consecutive
// threads walk consecutive k within one output channel.
#define GEMM_B_TPN (GEMM_THREADS / GEMM_BN)
#define GEMM_B_KPT (GEMM_BK / GEMM_B_TPN)

// Each thread's register tile is two half-tiles GEMM_BM/2 (resp. GEMM_BN/2)
// apart rather than one contiguous run: that is what keeps the shared-memory
// reads bank-conflict-free at 4 scalars per thread per half.
#define GEMM_TMH (GEMM_TM / 2)
#define GEMM_TNH (GEMM_TN / 2)

extern "C" __global__ __launch_bounds__(GEMM_THREADS) void cut_main(
    const cut_conv2d_t* __restrict__ input_data,
    const cut_conv2d_t* __restrict__ weight_data,
    cut_conv2d_t* __restrict__ output_data,
    PushConstants pc) {
    // Double buffered: the next k-tile is written while the current one is
    // still being read, so the loop needs one barrier per iteration.
    __shared__ __align__(16) cut_conv2d_t As[2][GEMM_BK][GEMM_BM];
    __shared__ __align__(16) cut_conv2d_t Bs[2][GEMM_BK][GEMM_BN];

    const uint H_out = (pc.H_in + 2 * pc.padH - pc.kH) / pc.strideH + 1;
    const uint W_out = (pc.W_in + 2 * pc.padW - pc.kW) / pc.strideW + 1;
    const uint inAlignedW = (pc.W_in + 3) & ~3u;
    const uint outAlignedW = (W_out + 3) & ~3u;
    const uint weightAlignedKW = (pc.kW + 3) & ~3u;

    const uint kHW = pc.kH * pc.kW;
    const uint K = pc.C_in * kHW;
    const uint HW = H_out * W_out;
    const uint M = pc.batchSize * HW;

    const uint tid = threadIdx.x;
    const uint mBase = blockIdx.x * GEMM_BM;
    const uint nBase = blockIdx.y * GEMM_BN;

    // --- A loader: one output position, GEMM_A_KPT consecutive k -----------
    const uint aM = tid % GEMM_BM;
    const uint aKOff = (tid / GEMM_BM) * GEMM_A_KPT;
    const uint aRow = mBase + aM;
    const bool aValid = aRow < M;
    const uint aChanStride = pc.H_in * inAlignedW;
    uint aBatchBase = 0;
    int aH0 = 0, aW0 = 0;
    {
        const uint m = aValid ? aRow : 0u;
        const uint nb = m / HW;
        const uint rem = m - nb * HW;
        const uint ho = rem / W_out;
        const uint wo = rem - ho * W_out;
        aBatchBase = nb * pc.C_in * aChanStride;
        // Top-left input pixel of this output position's window; the (kh, kw)
        // offset is added per k below.
        aH0 = int(ho * pc.strideH) - int(pc.padH);
        aW0 = int(wo * pc.strideW) - int(pc.padW);
    }

    // --- B loader: one output channel, GEMM_B_KPT consecutive k ------------
    const uint bN = tid / GEMM_B_TPN;
    const uint bKOff = (tid % GEMM_B_TPN) * GEMM_B_KPT;
    const uint bCol = nBase + bN;
    const bool bValid = bCol < pc.C_out;
    const uint bChanStride = pc.kH * weightAlignedKW;
    const uint bColBase = bValid ? bCol * pc.C_in * bChanStride : 0u;

    // --- k = (c_in, kh, kw), carried incrementally --------------------------
    // Decoding k costs two integer divisions, and doing that per element per
    // k-tile would cost a good fraction of the FMAs it feeds. Instead each
    // loader decodes its own k once and then advances by GEMM_BK per tile with
    // a precomputed (dci, dkh, dkw) delta and two carries — both deltas are
    // below their moduli, so one conditional subtraction each is exact.
    uint dci, dkh, dkw;
    {
        const uint c = GEMM_BK / kHW;
        const uint r = GEMM_BK - c * kHW;
        const uint h = r / pc.kW;
        dci = c;
        dkh = h;
        dkw = r - h * pc.kW;
    }
    uint aci, akh, akw, bci, bkh, bkw;
    {
        uint c = aKOff / kHW, r = aKOff - c * kHW, h = r / pc.kW;
        aci = c; akh = h; akw = r - h * pc.kW;
        c = bKOff / kHW; r = bKOff - c * kHW; h = r / pc.kW;
        bci = c; bkh = h; bkw = r - h * pc.kW;
    }

    cut_conv2d_t aReg[GEMM_A_KPT];
    cut_conv2d_t bReg[GEMM_B_KPT];

    // Gathers one k-tile into registers and advances both loaders' k triples.
    //
    // The nested-if (kw, kh, ci) walk looks like it wants to be branch-free
    // selects, and is not: written that way ptxas keeps every intermediate live
    // and the kernel goes 126 -> 143 registers, which is the difference between
    // two resident blocks per SM and one. Measured, not assumed.
    auto loadTile = [&](uint k0) {
        uint ci = aci, kh = akh, kw = akw;
#pragma unroll
        for (int i = 0; i < GEMM_A_KPT; ++i) {
            cut_conv2d_t v = (cut_conv2d_t)(0);
            if (aValid && k0 + aKOff + uint(i) < K) {
                const int ih = aH0 + int(kh);
                const int iw = aW0 + int(kw);
                if (ih >= 0 && ih < int(pc.H_in) && iw >= 0 && iw < int(pc.W_in))
                    v = input_data[aBatchBase + ci * aChanStride
                                   + uint(ih) * inAlignedW + uint(iw)];
            }
            aReg[i] = v;
            if (++kw == pc.kW) {
                kw = 0;
                if (++kh == pc.kH) { kh = 0; ++ci; }
            }
        }

        ci = bci; kh = bkh; kw = bkw;
#pragma unroll
        for (int i = 0; i < GEMM_B_KPT; ++i) {
            cut_conv2d_t v = (cut_conv2d_t)(0);
            if (bValid && k0 + bKOff + uint(i) < K)
                v = weight_data[bColBase + ci * bChanStride
                                + kh * weightAlignedKW + kw];
            bReg[i] = v;
            if (++kw == pc.kW) {
                kw = 0;
                if (++kh == pc.kH) { kh = 0; ++ci; }
            }
        }

        akw += dkw;
        const uint ac1 = (akw >= pc.kW) ? 1u : 0u;
        akw -= ac1 * pc.kW;
        akh += dkh + ac1;
        const uint ac2 = (akh >= pc.kH) ? 1u : 0u;
        akh -= ac2 * pc.kH;
        aci += dci + ac2;

        bkw += dkw;
        const uint bc1 = (bkw >= pc.kW) ? 1u : 0u;
        bkw -= bc1 * pc.kW;
        bkh += dkh + bc1;
        const uint bc2 = (bkh >= pc.kH) ? 1u : 0u;
        bkh -= bc2 * pc.kH;
        bci += dci + bc2;
    };

    // A is staged transposed (k-major) so the compute loop reads GEMM_TMH
    // contiguous scalars per thread.
    auto storeTile = [&](uint buf) {
#pragma unroll
        for (int i = 0; i < GEMM_A_KPT; ++i)
            As[buf][aKOff + uint(i)][aM] = aReg[i];
#pragma unroll
        for (int i = 0; i < GEMM_B_KPT; ++i)
            Bs[buf][bKOff + uint(i)][bN] = bReg[i];
    };

    cut_conv2d_t acc[GEMM_TM][GEMM_TN];
#pragma unroll
    for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j)
            acc[i][j] = (cut_conv2d_t)(0);

    const uint tm = tid % (GEMM_BM / GEMM_TM);
    const uint tn = tid / (GEMM_BM / GEMM_TM);
    const uint aFrag0 = tm * GEMM_TMH;
    const uint bFrag0 = tn * GEMM_TNH;

    loadTile(0);
    storeTile(0);
    __syncthreads();

    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += GEMM_BK) {
        const bool hasNext = (k0 + GEMM_BK) < K;
        // Issued before the arithmetic so the global latency hides under it.
        if (hasNext)
            loadTile(k0 + GEMM_BK);

#pragma unroll
        for (int kk = 0; kk < GEMM_BK; ++kk) {
            cut_conv2d_t af[GEMM_TM], bf[GEMM_TN];
#pragma unroll
            for (int i = 0; i < GEMM_TMH; ++i) {
                af[i] = As[buf][kk][aFrag0 + uint(i)];
                af[i + GEMM_TMH] = As[buf][kk][GEMM_BM / 2 + aFrag0 + uint(i)];
            }
#pragma unroll
            for (int j = 0; j < GEMM_TNH; ++j) {
                bf[j] = Bs[buf][kk][bFrag0 + uint(j)];
                bf[j + GEMM_TNH] = Bs[buf][kk][GEMM_BN / 2 + bFrag0 + uint(j)];
            }
#pragma unroll
            for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
                for (int j = 0; j < GEMM_TN; ++j)
                    acc[i][j] += af[i] * bf[j];
        }

        if (hasNext)
            storeTile(buf ^ 1u);
        __syncthreads();
        buf ^= 1u;
    }

    // --- Scatter the register tile back to NCHW -----------------------------
    const uint outChanStride = H_out * outAlignedW;
#pragma unroll
    for (int i = 0; i < GEMM_TM; ++i) {
        const uint mo = mBase + ((i < GEMM_TMH)
                                     ? (aFrag0 + uint(i))
                                     : (GEMM_BM / 2 + aFrag0 + uint(i - GEMM_TMH)));
        if (mo >= M)
            continue;
        const uint nb = mo / HW;
        const uint rem = mo - nb * HW;
        const uint ho = rem / W_out;
        const uint wo = rem - ho * W_out;
        const uint rowBase = nb * pc.C_out * outChanStride + ho * outAlignedW + wo;
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) {
            const uint co = nBase + ((j < GEMM_TNH)
                                         ? (bFrag0 + uint(j))
                                         : (GEMM_BN / 2 + bFrag0 + uint(j - GEMM_TNH)));
            if (co < pc.C_out)
                output_data[rowBase + co * outChanStride] = acc[i][j];
        }
    }
}
