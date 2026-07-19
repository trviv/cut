// Native CUDA counterpart of RadixScatter.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

#define RADIX 16

struct PushConstants {
    uint numElements;
    uint bitOffset;
    uint groupCount;
};

extern "C" __global__ void cut_main(const uint* __restrict__ keysIn,
                                    const uint* __restrict__ valsIn,
                                    uint* __restrict__ keysOut,
                                    uint* __restrict__ valsOut,
                                    uint* __restrict__ scannedHist,
                                    PushConstants pc) {
    // Load global starting offsets for each digit
    uint offset[RADIX];
#pragma unroll
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
