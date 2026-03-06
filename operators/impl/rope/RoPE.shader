#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint numElements; // n_heads * head_dim
    uint headDim;
    uint halfDim;     // head_dim / 2
};
[[vk::push_constant]] PushConstants pc;

// Input tensor [numElements]
[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
// Precomputed cos table [maxSeqLen * halfDim]
[[vk::binding(1, 0)]] StructuredBuffer<float> cosTable;
// Precomputed sin table [maxSeqLen * halfDim]
[[vk::binding(2, 0)]] StructuredBuffer<float> sinTable;
// Runtime params [pos, seqLen]
[[vk::binding(3, 0)]] StructuredBuffer<uint> runtimeParams;
// Output tensor [numElements]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint gid = DTid.x;
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
