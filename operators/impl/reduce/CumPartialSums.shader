#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256

struct PushConstants {
    uint groupsPerLine;
    uint numScanLines;
    uint cumOp;  // 0 = sum, 1 = prod
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> partialSums;

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint lineIdx = DTid.x;
    if (lineIdx >= pc.numScanLines) {
        return;
    }

    uint base = lineIdx * pc.groupsPerLine;

    // Exclusive prefix scan of workgroup totals for this scan line
    %SCALAR_DTYPE_INPUT% acc;
    if (pc.cumOp == 0u) {
        acc = (%SCALAR_DTYPE_INPUT%)(0);
    } else {
        acc = (%SCALAR_DTYPE_INPUT%)(1);
    }

    for (uint i = 0; i < pc.groupsPerLine; i++) {
        %SCALAR_DTYPE_INPUT% val = partialSums[base + i];
        partialSums[base + i] = acc;
        if (pc.cumOp == 0u) {
            acc += val;
        } else {
            acc *= val;
        }
    }
}
