// Native CUDA counterpart of RoPEInterleaved.shader — keep semantics in lockstep.
//
// Interleaved-pair rotary embedding (LTX/DiT convention):
//   out[2k]   = x[2k]   * cos[2k]   - x[2k+1] * sin[2k]
//   out[2k+1] = x[2k+1] * cos[2k+1] + x[2k]   * sin[2k+1]
// cos/sin tables have the same shape as x (precomputed per position on host).
#include "ComputeOpsShared.h"

struct PushConstants {
    uint numElements;
};

extern "C" __global__ void cut_main(const float* __restrict__ dataIn,
                                    const float* __restrict__ cosTable,
                                    const float* __restrict__ sinTable,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pc.numElements) return;
    float x = dataIn[i];
    float rot = ((i & 1u) == 0u) ? -dataIn[i + 1] : dataIn[i - 1];
    dataOut[i] = x * cosTable[i] + rot * sinTable[i];
}
