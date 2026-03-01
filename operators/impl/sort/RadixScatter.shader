#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define RADIX 16

struct PushConstants {
    uint numElements;
    uint bitOffset;
    uint groupCount;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> keysIn;
[[vk::binding(1, 0)]] StructuredBuffer<uint> valsIn;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> keysOut;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> valsOut;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> scannedHist;

[numthreads(1, 1, 1)]
void main() {
    // Load global starting offsets for each digit
    uint offset[RADIX];
    for (uint d = 0; d < RADIX; d++) {
        offset[d] = scannedHist[d * pc.groupCount];
    }

    // Sequential scatter preserves input order (stability)
    for (uint i = 0; i < pc.numElements; i++) {
        uint key = keysIn[i];
        uint digit = (key >> pc.bitOffset) & 0xFu;
        uint pos = offset[digit];
        keysOut[pos] = key;
        valsOut[pos] = valsIn[i];
        offset[digit] = pos + 1;
    }
}
