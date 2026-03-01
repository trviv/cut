#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

[[vk::constant_id(0)]] const uint dtype_vec_size = 4;

struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT%> dataCond;
[[vk::binding(1, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT%> dataX;
[[vk::binding(2, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT%> dataY;
[[vk::binding(3, 0)]] RWStructuredBuffer<%VEC_DTYPE_INPUT%> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;
    if (index * dtype_vec_size >= pc.numElements) {
        return;
    }

    // Select from X where condition is non-zero, otherwise from Y
    %VEC_DTYPE_INPUT% cond = dataCond[index];
    dataOut[index] = select(cond != (%VEC_DTYPE_INPUT%)(0), dataX[index], dataY[index]);
}
