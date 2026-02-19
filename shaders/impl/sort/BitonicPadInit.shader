#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

struct PushConstants {
    uint numElements;
    uint paddedSize;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> srcKeys;
[[vk::binding(1, 0)]] StructuredBuffer<uint> srcVals;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dstKeys;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> dstVals;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint idx = DTid.x;
    if (idx >= pc.paddedSize) return;
    if (idx < pc.numElements) {
        dstKeys[idx] = srcKeys[idx];
        dstVals[idx] = srcVals[idx];
    } else {
        dstKeys[idx] = 3.402823466e+38;
        dstVals[idx] = 0xFFFFFFFFu;
    }
}
