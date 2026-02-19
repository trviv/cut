#include "ComputeOpsShared.h"

// Specialization constants
[[vk::constant_id(1)]] const uint op_enum = OP_REDUCE_ARGMAX;

struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;

[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

groupshared float sharedVal[256];
groupshared uint  sharedIdx[256];

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
void main(uint3 GTid : SV_GroupThreadID) {
    uint tid = GTid.x;

    float localVal = worstVal();
    uint localIdx = 0;
    for (uint i = tid; i < pc.numElements; i += 256) {
        float b = dataIn[i];
        if (isBetter(b, localVal)) {
            localVal = b;
            localIdx = i;
        }
    }
    sharedVal[tid] = localVal;
    sharedIdx[tid] = localIdx;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (isBetter(sharedVal[tid + stride], sharedVal[tid])) {
                sharedVal[tid] = sharedVal[tid + stride];
                sharedIdx[tid] = sharedIdx[tid + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        dataOut[0] = (float)(sharedIdx[0]);
    }
}
