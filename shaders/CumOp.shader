#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Specialization constants
layout(constant_id = 1) const uint op_enum = OP_CUMSUM;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint outerSize;
    uint reduceSize;
    uint innerSize;
    uint inOuterStride;
    uint inReduceStride;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %SCALAR_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %SCALAR_DTYPE% dataOut[];
};

void main() {
    uint outIdx = gl_GlobalInvocationID.x;
    uint numScanLines = outerSize * innerSize;

    if (outIdx >= numScanLines) {
        return;
    }

    uint outer = outIdx / innerSize;
    uint inner = outIdx % innerSize;

    %SCALAR_DTYPE% acc;
    if (op_enum == OP_CUMSUM) {
        acc = %SCALAR_DTYPE%(0);
    } else {
        acc = %SCALAR_DTYPE%(1);
    }

    for (uint r = 0; r < reduceSize; r++) {
        uint idx = outer * inOuterStride + r * inReduceStride + inner;
        if (op_enum == OP_CUMSUM) {
            acc += dataIn[idx];
        } else {
            acc *= dataIn[idx];
        }
        dataOut[idx] = acc;
    }
}
