#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Specialization constants
[[vk::constant_id(0)]] const uint dtype_vec_size = %DTYPE_SIZE_INPUT%;
[[vk::constant_id(1)]] const uint op_enum = OP_BINARY_EQUAL;

// Push constants
struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

// Storage buffers
[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT%> dataA;
[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE_INPUT%> dataB;
[[vk::binding(2, 0)]] RWStructuredBuffer<%VEC_DTYPE_OUTPUT%> dataOut;

#include "BinaryCmpOps.shaderh"

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;

    if (index * dtype_vec_size >= pc.numElements) {
        return;
    }

    %VEC_DTYPE_INPUT% a = dataA[index];
    %VEC_DTYPE_INPUT% s = (%VEC_DTYPE_INPUT%)(dataB[0]);

    dataOut[index] = binaryCmpOp(a, s);
}
