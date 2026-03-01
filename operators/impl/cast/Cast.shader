#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%
%DTYPE_DEFINES_OUTPUT%

struct PushConstants {
    uint alignedInner;
    uint actualInner;
    uint totalElements;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_OUTPUT%> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint gid = DTid.x;

    if (gid >= pc.totalElements) {
        return;
    }

    uint row = gid / pc.actualInner;
    uint col = gid % pc.actualInner;
    uint idx = row * pc.alignedInner + col;

    dataOut[idx] = (%SCALAR_DTYPE_OUTPUT%)(dataIn[idx]);
}
