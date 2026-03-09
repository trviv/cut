#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%
%DTYPE_DEFINES_OUTPUT%

struct PushConstants {
    uint numElements;
};
[[vk::push_constant]] PushConstants pc;

// logits [N] Float32
[[vk::binding(0, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT%> logits;
// penaltyFactors [N] Float32 (1.0 = no penalty, >1.0 = penalized)
[[vk::binding(1, 0)]] StructuredBuffer<%VEC_DTYPE_INPUT%> factors;
// output [N] Float32
[[vk::binding(2, 0)]] RWStructuredBuffer<%VEC_DTYPE_OUTPUT%> dataOut;

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;
    uint baseElem = index * 4;
    if (baseElem >= pc.numElements) return;

    %VEC_DTYPE_INPUT% l = logits[index];
    %VEC_DTYPE_INPUT% f = factors[index];

    // Per-component: logit > 0 ? logit/factor : logit*factor
    // factor == 1.0 for non-penalized tokens → identity operation
    %VEC_DTYPE_OUTPUT% result;
    result.x = l.x > 0.0 ? l.x / f.x : l.x * f.x;
    result.y = l.y > 0.0 ? l.y / f.y : l.y * f.y;
    result.z = l.z > 0.0 ? l.z / f.z : l.z * f.z;
    result.w = l.w > 0.0 ? l.w / f.w : l.w * f.w;
    dataOut[index] = result;
}
