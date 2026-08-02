// Native CUDA softmax. The algorithm — warp-or-block per row, vectorized,
// register-resident when the row fits — lives in SoftmaxCommon.cuh, which also
// documents how and why it diverges from Softmax.shader.
#include "ComputeOpsShared.h"
#include "SoftmaxCommon.cuh"

extern "C" __global__ void cut_main(const CUT_SCALAR_DTYPE_INPUT* __restrict__ dataIn,
                                    CUT_SCALAR_DTYPE_INPUT* __restrict__ dataOut,
                                    PushConstants pc) {
    cut_softmax_kernel<false>(dataIn, dataOut, pc);
}
