#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

#define WG_SIZE 256
#define ELEMS_PER_THREAD 8
#define TILE_SIZE (WG_SIZE * ELEMS_PER_THREAD)

struct PushConstants {
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
    uint groupsPerLine;
    uint cumOp;  // 0 = sum, 1 = prod
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> partialSums;
[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> dataOut;

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;
    uint wgIdx = Gid.x;   // workgroup index along reduce dimension
    uint lineIdx = Gid.y;  // scan line index

    // Skip workgroup 0 — its prefix is identity (0 for sum, 1 for prod)
    if (wgIdx == 0) {
        return;
    }

    uint outer = lineIdx / pc.innerSize;
    uint inner = lineIdx % pc.innerSize;
    uint baseOffset = outer * pc.inOuterStride + inner;

    %SCALAR_DTYPE_INPUT% prefix = partialSums[lineIdx * pc.groupsPerLine + wgIdx];

    uint tileStart = wgIdx * TILE_SIZE;

    for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
        uint r = tileStart + tid * ELEMS_PER_THREAD + e;
        if (r < pc.reduceSize) {
            uint idx = baseOffset + r * pc.inReduceStride;
            if (pc.cumOp == 0u) {
                dataOut[idx] += prefix;
            } else {
                dataOut[idx] *= prefix;
            }
        }
    }
}
