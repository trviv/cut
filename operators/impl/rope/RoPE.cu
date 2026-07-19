// Native CUDA counterpart of RoPE.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

struct PushConstants {
    uint numElements; // n_heads * head_dim
    uint headDim;
    uint halfDim;     // head_dim / 2
};

extern "C" __global__ void cut_main(const float* __restrict__ dataIn,
                                    const float* __restrict__ cosTable,
                                    const float* __restrict__ sinTable,
                                    const uint* __restrict__ runtimeParams,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pc.numElements) return;

    uint pos = runtimeParams[0];
    uint idxInHead = gid % pc.headDim;
    uint tableBase = pos * pc.halfDim;

    if (idxInHead < pc.halfDim) {
        // First half: out = x[i] * cos - x[i + halfDim] * sin
        float cosVal = cosTable[tableBase + idxInHead];
        float sinVal = sinTable[tableBase + idxInHead];
        float x0 = dataIn[gid];
        float x1 = dataIn[gid + pc.halfDim];
        dataOut[gid] = x0 * cosVal - x1 * sinVal;
    } else {
        // Second half: out = x[i - halfDim] * sin + x[i] * cos
        uint pairIdx = idxInHead - pc.halfDim;
        float cosVal = cosTable[tableBase + pairIdx];
        float sinVal = sinTable[tableBase + pairIdx];
        float x0 = dataIn[gid - pc.halfDim];
        float x1 = dataIn[gid];
        dataOut[gid] = x0 * sinVal + x1 * cosVal;
    }
}
