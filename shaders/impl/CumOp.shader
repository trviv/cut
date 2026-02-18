#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Specialization constants
[[vk::constant_id(1)]] const uint op_enum = OP_CUMSUM;

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

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint outIdx = DTid.x;
    uint numScanLines = pc.outerSize * pc.innerSize;

    if (outIdx >= numScanLines) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    %SCALAR_DTYPE% acc;
    if (op_enum == OP_CUMSUM) {
        acc = (%SCALAR_DTYPE%)(0);
    } else {
        acc = (%SCALAR_DTYPE%)(1);
    }

    for (uint r = 0; r < pc.reduceSize; r++) {
        uint idx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        if (op_enum == OP_CUMSUM) {
            acc += dataIn[idx];
        } else {
            acc *= dataIn[idx];
        }
        dataOut[idx] = acc;
    }
}
