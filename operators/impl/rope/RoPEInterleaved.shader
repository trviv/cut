#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// Interleaved-pair rotary embedding (LTX/DiT convention):
//   out[2k]   = x[2k]   * cos[2k]   - x[2k+1] * sin[2k]
//   out[2k+1] = x[2k+1] * cos[2k+1] + x[2k]   * sin[2k+1]
// cos/sin tables have the same shape as x (precomputed per position on host).
struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] StructuredBuffer<float> cosTable;
[[vk::binding(2, 0)]] StructuredBuffer<float> sinTable;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint i = DTid.x;
    if (i >= pc.numElements) return;
    float x = dataIn[i];
    float rot = ((i & 1u) == 0u) ? -dataIn[i + 1] : dataIn[i - 1];
    dataOut[i] = x * cosTable[i] + rot * sinTable[i];
}
