#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

[[vk::constant_id(0)]] const uint dtype_vec_size = 4;

struct PushConstants {
    uint numElements;
    %SCALAR_DTYPE_INPUT% fillValue;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<%VEC_DTYPE_INPUT%> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;
    if (index * dtype_vec_size >= pc.numElements) {
        return;
    }

    dataOut[index] = (%VEC_DTYPE_INPUT%)(pc.fillValue);
}
