#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

// Specialization constants
[[vk::constant_id(0)]] const uint dtype_vec_size = %DTYPE_SIZE%;
[[vk::constant_id(1)]] const uint op_enum = OP_BINARY_ADD;

// Push constants
struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

// Storage buffers
[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE%> dataA;
[[vk::binding(1, 0)]] StructuredBuffer<%VEC_DTYPE%> dataB;
[[vk::binding(2, 0)]] RWStructuredBuffer<%VEC_DTYPE%> dataOut;

#include "BinaryOps.shaderh"

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;

    if (index * dtype_vec_size >= pc.numElements) {
        return;
    }

    dataOut[index] = binaryOp(dataA[index], dataB[index]);
}
