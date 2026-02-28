#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

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

#include "ReduceDimCommon.shaderh"

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint outIdx = DTid.x;
    uint numOutputs = pc.outerSize * pc.innerSize;

    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    %SCALAR_DTYPE% val = identity();
    for (uint r = 0; r < pc.reduceSize; r++) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        val = reduceOp(val, dataIn[inIdx]);
    }

    dataOut[outIdx] = finalizeReduce(val, pc.reduceSize);
}
