// Native CUDA counterpart of MatMulCoopMat.comp (KHR cooperative matrix, 16x16 tile per warp).
// Tensor-core MMA via inline-PTX helpers in MatMulCoopMatCommon.cuh (mma.h is unusable under NVRTC).
#include "MatMulCoopMatCommon.cuh"

extern "C" __global__ void cut_main(const half* __restrict__ dataA,
                                    const half* __restrict__ dataB,
                                    const float* __restrict__ dataD,
                                    float* __restrict__ dataC,
                                    CoopMatPush pc) {
    uint tileRow = blockIdx.y * 16u;
    uint tileCol = blockIdx.x * 16u;
    if (tileRow >= pc.M || tileCol >= pc.N) return;

    int lane = threadIdx.x & 31;

    Acc16 acc;
    zeroAcc16(acc);

    for (uint k = 0; k < pc.K; k += 16u) {
        FragA16 a;
        cutLoadA16(a, dataA + tileRow * pc.strideA + k, (int)pc.strideA, lane);
        FragB16 b;
        cutLoadB16(b, dataB + k * pc.strideB + tileCol, (int)pc.strideB, lane);
        cutMma16(acc, a, b);
    }

    cutStoreC16(dataC + tileRow * pc.strideC + tileCol, (int)pc.strideC, acc, lane);
}
