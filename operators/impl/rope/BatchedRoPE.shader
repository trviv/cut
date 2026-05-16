#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Apply RoPE to all N tokens of a batch in one dispatch.
// Reads from a (possibly strided) source buffer: row i, column inRowOffset+j
// of the [N, inRowStride] input. Writes to a contiguous [N, dim] output
// buffer (row stride = alignedDim).
struct PushConstants {
    uint batchSize;
    uint dim;
    uint alignedDim;
    uint inRowStride;
    uint inRowOffset;
    uint headDim;
    uint halfDim;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] StructuredBuffer<float> cosTable;
[[vk::binding(2, 0)]] StructuredBuffer<float> sinTable;
[[vk::binding(3, 0)]] StructuredBuffer<uint> positions;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint gid = Gid.x * 256 + GTid.x;
    uint tokenIdx = Gid.y;
    if (tokenIdx >= pc.batchSize) return;
    if (gid >= pc.dim) return;

    uint pos = positions[tokenIdx];
    uint inOff  = tokenIdx * pc.inRowStride + pc.inRowOffset;
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
