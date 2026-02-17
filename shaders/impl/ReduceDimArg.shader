#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

// Specialization constants
layout(constant_id = 1) const uint op_enum = OP_REDUCE_ARGMAX;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

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

void main() {
    uint outIdx = gl_GlobalInvocationID.x;
    uint numOutputs = outerSize * innerSize;

    if (outIdx >= numOutputs) {
        return;
    }

    uint outer = outIdx / innerSize;
    uint inner = outIdx % innerSize;

    float bestVal = worstVal();
    uint bestIdx = 0;
    for (uint r = 0; r < reduceSize; r++) {
        uint inIdx = outer * inOuterStride + r * inReduceStride + inner;
        float b = dataIn[inIdx];
        if (isBetter(b, bestVal)) {
            bestVal = b;
            bestIdx = r;
        }
    }
    dataOut[outIdx] = float(bestIdx);
}
