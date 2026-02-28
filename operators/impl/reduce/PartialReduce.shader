#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

struct PushConstants {
    uint numElements;
    uint groupCount;
    uint reduceOp;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataOut;

groupshared %SCALAR_DTYPE% sharedData[256];

#define REDUCE_OP_VAR pc.reduceOp
#include "ReduceCommon.shaderh"

[numthreads(256, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint tid = GTid.x;
    uint gid = Gid.x;

    %SCALAR_DTYPE% localVal = identity();
    for (uint i = gid * 256 + tid; i < pc.numElements; i += 256 * pc.groupCount) {
        localVal = reduceOp(localVal, dataIn[i]);
    }
    sharedData[tid] = localVal;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = reduceOp(sharedData[tid], sharedData[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        dataOut[gid] = sharedData[0];
    }
}
