#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

// Specialization constants
layout(constant_id = 1) const uint op_enum = OP_REDUCE_ARGMAX;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    float dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    float dataOut[];
};

shared float sharedVal[256];
shared uint  sharedIdx[256];

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
    uint tid = gl_LocalInvocationID.x;

    float localVal = worstVal();
    uint localIdx = 0;
    for (uint i = tid; i < numElements; i += 256) {
        float b = dataIn[i];
        if (isBetter(b, localVal)) {
            localVal = b;
            localIdx = i;
        }
    }
    sharedVal[tid] = localVal;
    sharedIdx[tid] = localIdx;
    barrier();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (isBetter(sharedVal[tid + stride], sharedVal[tid])) {
                sharedVal[tid] = sharedVal[tid + stride];
                sharedIdx[tid] = sharedIdx[tid + stride];
            }
        }
        barrier();
    }

    if (tid == 0) {
        dataOut[0] = float(sharedIdx[0]);
    }
}
