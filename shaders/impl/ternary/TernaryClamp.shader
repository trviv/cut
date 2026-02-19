#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

[[vk::constant_id(0)]] const uint dtype_vec_size = 4;

struct PushConstants {
    uint numElements;
    %SCALAR_DTYPE% minVal;
    %SCALAR_DTYPE% maxVal;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE%> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<%VEC_DTYPE%> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;
    if (index * dtype_vec_size >= pc.numElements) {
        return;
    }

    dataOut[index] = clamp(dataIn[index], (%VEC_DTYPE%)(pc.minVal), (%VEC_DTYPE%)(pc.maxVal));
}
