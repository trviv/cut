#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

[[vk::constant_id(0)]] const uint dtype_vec_size = 4;

struct PushConstants {
    uint numElements;
    %SCALAR_DTYPE% start;
    %SCALAR_DTYPE% step;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<%VEC_DTYPE%> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;
    uint baseIdx = index * dtype_vec_size;
    if (baseIdx >= pc.numElements) {
        return;
    }

    // Generate 4 consecutive values
    %VEC_DTYPE% result;
    for (uint i = 0; i < dtype_vec_size && (baseIdx + i) < pc.numElements; i++) {
        result[i] = pc.start + (%SCALAR_DTYPE%)(baseIdx + i) * pc.step;
    }
    dataOut[index] = result;
}
