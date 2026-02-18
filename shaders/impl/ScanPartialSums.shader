#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

struct PushConstants {
    uint numGroups;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> partialSums;

[numthreads(1, 1, 1)]
void main() {
    float sum = 0.0;
    for (uint i = 0; i < pc.numGroups; i++) {
        float val = partialSums[i];
        partialSums[i] = sum;
        sum += val;
    }
}
