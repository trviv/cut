// Native CUDA counterpart of MatMulQ4Gemv.shader. Semantics mirror the transpiled reference.
#include "MatMulQ4Common.cuh"

extern "C" __global__ void cut_main(const CUT_VEC_DTYPE_INPUT1* __restrict__ dataA,
                                    const uint* __restrict__ packedB,
                                    const CUT_VEC_DTYPE_SCALES* __restrict__ scalesB,
                                    const CUT_VEC_DTYPE_OUTPUT* __restrict__ dataD,
                                    CUT_SCALAR_DTYPE_OUTPUT* __restrict__ dataC,
                                    PushConstants pc) {
    uint n = blockIdx.x * blockDim.x + threadIdx.x;
    uint m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= pc.N || m >= pc.M) return;

    float acc = 0.0f;
    uint K4 = pc.K & ~3u;
    uint k = 0;
    for (; k < K4; k += 4) {
        acc = mad(float(cut_loadA(dataA, pc, m, k)),     cut_loadB(packedB, scalesB, pc, k,     n), acc);
        acc = mad(float(cut_loadA(dataA, pc, m, k + 1)), cut_loadB(packedB, scalesB, pc, k + 1, n), acc);
        acc = mad(float(cut_loadA(dataA, pc, m, k + 2)), cut_loadB(packedB, scalesB, pc, k + 2, n), acc);
        acc = mad(float(cut_loadA(dataA, pc, m, k + 3)), cut_loadB(packedB, scalesB, pc, k + 3, n), acc);
    }
    for (; k < pc.K; k++) {
        acc = mad(float(cut_loadA(dataA, pc, m, k)), cut_loadB(packedB, scalesB, pc, k, n), acc);
    }
    cut_writeOutput(dataC, dataD, pc, m, n, acc);
}
