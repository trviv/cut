// Native CUDA counterpart of BatchedRoPE.shader — keep semantics in lockstep.
//
// Apply RoPE to all N tokens of a batch in one dispatch. Reads from a
// (possibly strided) source buffer: row i, column inRowOffset+j of the
// [N, inRowStride] input. Writes to a contiguous [N, dim] output buffer
// (row stride = alignedDim).
#include "ComputeOpsShared.h"

struct PushConstants {
    uint batchSize;
    uint dim;
    uint alignedDim;
    uint inRowStride;
    uint inRowOffset;
    uint headDim;
    uint halfDim;
};

extern "C" __global__ void cut_main(const float* __restrict__ dataIn,
                                    const float* __restrict__ cosTable,
                                    const float* __restrict__ sinTable,
                                    const uint* __restrict__ positions,
                                    float* __restrict__ dataOut,
                                    PushConstants pc) {
    uint gid = blockIdx.x * 256u + threadIdx.x;
    uint tokenIdx = blockIdx.y;
    if (tokenIdx >= pc.batchSize) return;
    if (gid >= pc.dim) return;

    uint pos = positions[tokenIdx];
    uint inOff = tokenIdx * pc.inRowStride + pc.inRowOffset;
    uint outOff = tokenIdx * pc.alignedDim;
    uint idxInHead = gid % pc.headDim;
    uint tableBase = pos * pc.halfDim;

    if (idxInHead < pc.halfDim) {
        float cosVal = cosTable[tableBase + idxInHead];
        float sinVal = sinTable[tableBase + idxInHead];
        float x0 = dataIn[inOff + gid];
        float x1 = dataIn[inOff + gid + pc.halfDim];
        dataOut[outOff + gid] = x0 * cosVal - x1 * sinVal;
    } else {
        uint pairIdx = idxInHead - pc.halfDim;
        float cosVal = cosTable[tableBase + pairIdx];
        float sinVal = sinTable[tableBase + pairIdx];
        float x0 = dataIn[inOff + gid - pc.halfDim];
        float x1 = dataIn[inOff + gid];
        dataOut[outOff + gid] = x0 * sinVal + x1 * cosVal;
    }
}
