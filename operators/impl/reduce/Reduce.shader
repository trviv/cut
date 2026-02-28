#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Specialization constants
[[vk::constant_id(1)]] const uint op_enum = OP_REDUCE_SUM;

struct PushConstants {
    uint numElements;
    uint actualInner;
    uint alignedInner;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataOut;

groupshared %SCALAR_DTYPE% sharedData[256];

#include "ReduceCommon.shaderh"

[numthreads(256, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID) {
    uint tid = GTid.x;

    // Each thread reduces multiple elements via strided loop
    %SCALAR_DTYPE% localVal = identity();
    for (uint i = tid; i < pc.numElements; i += 256) {
        uint row = i / pc.actualInner;
        uint col = i % pc.actualInner;
        uint idx = row * pc.alignedInner + col;
        localVal = reduceOp(localVal, dataIn[idx]);
    }
    sharedData[tid] = localVal;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = reduceOp(sharedData[tid], sharedData[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Single workgroup: write result directly
    if (tid == 0) {
        %SCALAR_DTYPE% result = sharedData[0];
        // For mean, divide by total element count
        if (op_enum == OP_REDUCE_MEAN) {
            result = result / (%SCALAR_DTYPE%)(pc.numElements);
        }
        dataOut[0] = result;
    }
}
