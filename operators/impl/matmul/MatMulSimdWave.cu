// Native CUDA counterpart of MatMulSimdWave.shader (SIMD-wave matmul via warp shuffle).
#include "ComputeOpsShared.h"

#ifndef WR
#define WR 8
#endif
#ifndef WC
#define WC 4
#endif
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 4
#endif

#include "MatMulCommon.cuh"

extern "C" __global__ void cut_main(const cut_a_vec* __restrict__ dataA, const cut_b_vec* __restrict__ dataB, const cut_d_vec* __restrict__ dataD, cut_c_t* __restrict__ dataC, PushConstants pc) {
    uint3 GTid;
    GTid.x = threadIdx.x;
    GTid.y = threadIdx.y;
    uint3 Gid;
    Gid.x = blockIdx.x;
    Gid.y = blockIdx.y;

    uint laneId = GTid.x;
    uint wr = laneId / WC;
    uint wc = laneId % WC;

    uint blockRowStart = Gid.y * (WR * TM);
    uint blockColStart = Gid.x * (WC * TN);

    cut_c_t acc[TM][TN];
    #pragma unroll
    for (uint m = 0; m < TM; m++)
        #pragma unroll
        for (uint n = 0; n < TN; n++)
            acc[m][n] = (cut_c_t)(0);

    uint numKTiles = (pc.K + WC - 1) / WC;

    for (uint kt = 0; kt < numKTiles; kt++) {
        uint kBase = kt * WC;

        cut_a_t a_reg[TM];
        #pragma unroll
        for (uint m = 0; m < TM; m++) {
            uint aRow = blockRowStart + wr * TM + m;
            uint aCol = kBase + wc;
            a_reg[m] = mmLoadA(dataA, pc, aRow, aCol);
        }

        cut_b_t b_reg[TN];
        #pragma unroll
        for (uint n = 0; n < TN; n++) {
            uint bRow = kBase + wr;
            uint bCol = blockColStart + wc * TN + n;
            b_reg[n] = (wr < WC) ? mmLoadB(dataB, pc, bRow, bCol) : (cut_b_t)(0);
        }

        #pragma unroll
        for (uint kk = 0; kk < WC; kk++) {
            cut_a_t a_k[TM];
            #pragma unroll
            for (uint m = 0; m < TM; m++) {
                a_k[m] = WaveReadLaneAt(a_reg[m], wr * WC + kk);
            }

            cut_b_t b_k[TN];
            #pragma unroll
            for (uint n = 0; n < TN; n++) {
                b_k[n] = WaveReadLaneAt(b_reg[n], kk * WC + wc);
            }

            #pragma unroll
            for (uint m = 0; m < TM; m++)
                #pragma unroll
                for (uint n = 0; n < TN; n++)
                    acc[m][n] += (cut_c_t)(a_k[m]) * (cut_c_t)(b_k[n]);
        }
    }

    #pragma unroll
    for (uint m = 0; m < TM; m++) {
        #pragma unroll
        for (uint n = 0; n < TN; n++) {
            uint outRow = blockRowStart + wr * TM + m;
            uint outCol = blockColStart + wc * TN + n;
            if (outRow < pc.M && outCol < pc.N) {
                mmWriteOutput(dataC, dataD, pc, outRow, outCol, acc[m][n]);
            }
        }
    }
}
