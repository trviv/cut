#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Specialization constants
[[vk::constant_id(0)]] const uint dtype_vec_size = %DTYPE_SIZE_OUTPUT%;
[[vk::constant_id(1)]] const uint op_enum1 = OP_BINARY_ADD;
[[vk::constant_id(2)]] const uint op_enum2 = OP_BINARY_ADD;

// Push constants
struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

// Storage buffers
[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT1%> dataA;
[[vk::binding(1, 0)]] StructuredBuffer<%SCALAR_DTYPE_OUTPUT%> dataScalar;
[[vk::binding(2, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT2%> dataB;
[[vk::binding(3, 0)]] RWStructuredBuffer<%VEC_DTYPE_OUTPUT%> dataOut;

#include "FusedBinaryOps.shaderh"

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;

    if (index * dtype_vec_size >= pc.numElements) {
        return;
    }

    %VEC_DTYPE_OUTPUT% a = (%VEC_DTYPE_OUTPUT%)dataA[index];
    %VEC_DTYPE_OUTPUT% s = (%VEC_DTYPE_OUTPUT%)((%VEC_DTYPE_OUTPUT%)(dataScalar[0]));
    %VEC_DTYPE_OUTPUT% b = (%VEC_DTYPE_OUTPUT%)dataB[index];

    %VEC_DTYPE_OUTPUT% intermediate = applyOp(op_enum1, a, s);
    dataOut[index] = applyOp(op_enum2, intermediate, b);
}
