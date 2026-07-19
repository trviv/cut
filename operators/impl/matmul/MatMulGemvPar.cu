// Native CUDA counterpart of MatMulGemvPar.shader (minimal GEMV test kernel: writes zeros).
#include "ComputeOpsShared.h"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif
#ifndef COLS_PER_WG
#define COLS_PER_WG 4
#endif
#ifndef THREADS_PER_COL
#define THREADS_PER_COL 64
#endif

#include "MatMulCommon.cuh"

extern "C" __global__ void cut_main(const cut_a_vec* __restrict__ dataA, const cut_b_vec* __restrict__ dataB, const cut_d_vec* __restrict__ dataD, cut_c_t* __restrict__ dataC, PushConstants pc) {
    uint tid = threadIdx.x;
    uint m = blockIdx.y;
    uint colIdx = tid / THREADS_PER_COL;
    uint tidInCol = tid % THREADS_PER_COL;
    uint n = blockIdx.x * COLS_PER_WG + colIdx;

    if (tidInCol == 0 && n < pc.N && m < pc.M) {
        mmWriteOutput(dataC, dataD, pc, m, n, (cut_c_t)(0));
    }
}
