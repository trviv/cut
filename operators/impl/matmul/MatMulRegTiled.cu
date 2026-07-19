// Native CUDA counterpart of MatMulRegTiled.shader (register-tiled matmul, no shared memory).
#include "ComputeOpsShared.h"

#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 4
#endif

#include "MatMulCommon.cuh"

extern "C" __global__ void cut_main(const cut_a_vec* __restrict__ dataA, const cut_b_vec* __restrict__ dataB, const cut_d_vec* __restrict__ dataD, cut_c_t* __restrict__ dataC, PushConstants pc) {
    uint baseRow = (blockIdx.y * blockDim.y + threadIdx.y) * TM;
    uint baseCol = (blockIdx.x * blockDim.x + threadIdx.x) * TN;

    cut_c_t acc[TM][TN];
    #pragma unroll
    for (uint m = 0; m < TM; m++) {
        #pragma unroll
        for (uint n = 0; n < TN; n++) {
            acc[m][n] = (cut_c_t)(0);
        }
    }

    for (uint k = 0; k < pc.K; k++) {
        cut_a_t a[TM];
        cut_b_t b[TN];

        #pragma unroll
        for (uint m = 0; m < TM; m++) {
            a[m] = mmLoadA(dataA, pc, baseRow + m, k);
        }

        #pragma unroll
        for (uint n = 0; n < TN; n++) {
            b[n] = mmLoadB(dataB, pc, k, baseCol + n);
        }

        #pragma unroll
        for (uint m = 0; m < TM; m++) {
            #pragma unroll
            for (uint n = 0; n < TN; n++) {
                acc[m][n] += (cut_c_t)(a[m]) * (cut_c_t)(b[n]);
            }
        }
    }

    #pragma unroll
    for (uint m = 0; m < TM; m++) {
        #pragma unroll
        for (uint n = 0; n < TN; n++) {
            if (baseRow + m < pc.M && baseCol + n < pc.N) {
                mmWriteOutput(dataC, dataD, pc, baseRow + m, baseCol + n, acc[m][n]);
            }
        }
    }
}
