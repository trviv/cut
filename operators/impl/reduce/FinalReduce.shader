#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

struct PushConstants {
    uint numElements;
    uint originalNumElements;
    uint reduceOp;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE_INPUT%> dataOut;

groupshared %SCALAR_DTYPE_INPUT% sharedData[256];

#define REDUCE_OP_VAR pc.reduceOp
#include "ReduceCommon.shaderh"

[numthreads(256, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID) {
    uint tid = GTid.x;

    %SCALAR_DTYPE_INPUT% localVal = identity();
    for (uint i = tid; i < pc.numElements; i += 256) {
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
        %SCALAR_DTYPE_INPUT% result = sharedData[0];
        if (pc.reduceOp == OP_REDUCE_MEAN) {
            result = result / (%SCALAR_DTYPE_INPUT%)(pc.originalNumElements);
        }
        dataOut[0] = result;
    }
}
