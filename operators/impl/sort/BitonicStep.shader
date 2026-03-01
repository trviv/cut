#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint numElements;
    uint outerStep;
    uint innerStep;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> keys;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> vals;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint idx = DTid.x;
    uint ixj = idx ^ pc.innerStep;

    if (ixj <= idx || idx >= pc.numElements || ixj >= pc.numElements) {
        return;
    }

    bool ascending = ((idx & pc.outerStep) == 0);

    float keyI = keys[idx];
    float keyJ = keys[ixj];

    if ((ascending && keyI > keyJ) || (!ascending && keyI < keyJ)) {
        keys[idx] = keyJ;
        keys[ixj] = keyI;
        uint valI = vals[idx];
        uint valJ = vals[ixj];
        vals[idx] = valJ;
        vals[ixj] = valI;
    }
}
