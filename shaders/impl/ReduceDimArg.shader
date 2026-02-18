#include "ComputeOpsShared.h"

// Specialization constants
[[vk::constant_id(1)]] const uint op_enum = OP_REDUCE_ARGMAX;

struct PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

bool isBetter(float candidate, float current) {
    if (op_enum == OP_REDUCE_ARGMAX) {
        return candidate > current;
    } else {
        return candidate < current;
    }
}

float worstVal() {
    if (op_enum == OP_REDUCE_ARGMAX) {
        return -3.402823466e+38;
    } else {
        return 3.402823466e+38;
    }
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint outIdx = DTid.x;
    uint numOutputs = pc.outerSize * pc.innerSize;

    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / pc.innerSize;
    uint inner = outIdx % pc.innerSize;

    float bestVal = worstVal();
    uint bestIdx = 0;
    for (uint r = 0; r < pc.reduceSize; r++) {
        uint inIdx = outer * pc.inOuterStride + r * pc.inReduceStride + inner;
        float b = dataIn[inIdx];
        if (isBetter(b, bestVal)) {
            bestVal = b;
            bestIdx = r;
        }
    }
    dataOut[outIdx] = (float)(bestIdx);
}
