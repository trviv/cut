#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Shared memory parallel reduction along a dimension
// WG_SIZE=%WG_SIZE% threads cooperatively reduce one output element

#define WG_SIZE %WG_SIZE%

// Specialization constants
[[vk::constant_id(1)]] const uint op_enum = OP_REDUCE_SUM;

struct PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%SCALAR_DTYPE%> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<%SCALAR_DTYPE%> dataOut;

groupshared %SCALAR_DTYPE% sharedData[WG_SIZE];

#include "ReduceDimCommon.shaderh"

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint outIdx = Gid.x;
    uint tid = GTid.x;
    uint numOutputs = pc.outerSize * pc.innerSize;

    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    // Phase 1: Each thread accumulates a strided portion of the reduce dimension
    %SCALAR_DTYPE% val = identity();
    for (uint r = tid; r < pc.reduceSize; r += WG_SIZE) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        val = reduceOp(val, dataIn[inIdx]);
    }
    sharedData[tid] = val;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Tree reduction in shared memory
    for (uint stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = reduceOp(sharedData[tid], sharedData[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Phase 3: Write result
    if (tid == 0) {
        dataOut[outIdx] = finalizeReduce(sharedData[0], pc.reduceSize);
    }
}
