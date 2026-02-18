#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

struct PushConstants {
    uint srcAlignedInner;
    uint srcActualInner;
    uint dstAlignedInner;
    uint dstActualInner;
    uint totalElements;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint gid = DTid.x;

    if (gid >= pc.totalElements) {
        return;
    }

    uint srcRow = gid / pc.srcActualInner;
    uint srcCol = gid % pc.srcActualInner;
    uint dstRow = gid / pc.dstActualInner;
    uint dstCol = gid % pc.dstActualInner;

    dataOut[dstRow * pc.dstAlignedInner + dstCol] = dataIn[srcRow * pc.srcAlignedInner + srcCol];
}
