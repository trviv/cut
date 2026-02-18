#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

#define WG_SIZE 256

struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> partialSums;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;
    uint gid = Gid.x;
    uint idx = gid * WG_SIZE + tid;

    if (idx < pc.numElements && gid > 0) {
        dataOut[idx] += partialSums[gid];
    }
}
