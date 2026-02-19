#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> data;

[numthreads(1, 1, 1)]
void main() {
    uint sum = 0;
    for (uint i = 0; i < pc.numElements; i++) {
        uint val = data[i];
        data[i] = sum;
        sum += val;
    }
}
