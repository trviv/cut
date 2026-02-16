#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

#define RADIX 16
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
    uint bitOffset;
    uint groupCount;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer KeysIn {
    uint keysIn[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer ValsIn {
    uint valsIn[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer KeysOut {
    uint keysOut[];
};

layout(set = 0, binding = 3, std430) restrict writeonly buffer ValsOut {
    uint valsOut[];
};

layout(set = 0, binding = 4, std430) restrict buffer ScannedHist {
    uint scannedHist[];
};

void main() {
    // Load global starting offsets for each digit
    uint offset[RADIX];
    for (uint d = 0; d < RADIX; d++) {
        offset[d] = scannedHist[d * groupCount];
    }

    // Sequential scatter preserves input order (stability)
    for (uint i = 0; i < numElements; i++) {
        uint key = keysIn[i];
        uint digit = (key >> bitOffset) & 0xFu;
        uint pos = offset[digit];
        keysOut[pos] = key;
        valsOut[pos] = valsIn[i];
        offset[digit] = pos + 1;
    }
}
