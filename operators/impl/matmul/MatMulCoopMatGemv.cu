// Native CUDA counterpart of MatMulCoopMatGemv.comp (M=1 GEMV, zero-padded A tile through tensor cores).
#include "MatMulCoopMatCommon.cuh"

extern "C" __global__ void cut_main(const half* __restrict__ dataA,
                                    const half* __restrict__ dataB,
                                    const float* __restrict__ dataD,
                                    float* __restrict__ dataC,
                                    CoopMatPush pc) {
    __shared__ __align__(16) half tileA[16 * 16];
    __shared__ float tileC[16 * 16];

    uint tileCol = blockIdx.x * 16u;
    if (tileCol >= pc.N) return;

    uint tid = threadIdx.x;   // 0..31
    int lane = (int)tid;

    Acc16 acc;
    zeroAcc16(acc);

    for (uint k = 0; k < pc.K; k += 16u) {
        for (uint i = tid; i < 256u; i += 32u)
            tileA[i] = (half)0.0f;
        if (tid < 16u)
            tileA[tid] = dataA[k + tid];
        __syncthreads();

        FragA16 a;
        cutLoadA16(a, tileA, 16, lane);
        FragB16 b;
        cutLoadB16(b, dataB + k * pc.strideB + tileCol, (int)pc.strideB, lane);
        cutMma16(acc, a, b);
        __syncthreads();
    }

    cutStoreC16(tileC, 16, acc, lane);
    __syncthreads();

    if (tid < 16u && (tileCol + tid) < pc.N)
        dataC[tileCol + tid] = tileC[tid];
}
