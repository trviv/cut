// Native CUDA counterpart of RepetitionPenalty.shader — keep semantics in lockstep.
#include "ComputeOpsShared.h"

#ifndef CUT_VEC_DTYPE_INPUT
#define CUT_VEC_DTYPE_INPUT float4
#endif
#ifndef CUT_VEC_DTYPE_OUTPUT
#define CUT_VEC_DTYPE_OUTPUT float4
#endif

typedef CUT_VEC_DTYPE_INPUT vec_in_t;
typedef CUT_VEC_DTYPE_OUTPUT vec_out_t;

struct PushConstants {
    uint numElements;
};

extern "C" __global__ void cut_main(const vec_in_t* __restrict__ logits,
                                    const vec_in_t* __restrict__ factors,
                                    vec_out_t* __restrict__ dataOut,
                                    PushConstants pc) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint baseElem = index * 4;
    if (baseElem >= pc.numElements) return;

    vec_in_t l = logits[index];
    vec_in_t f = factors[index];

    // Per-component: logit > 0 ? logit/factor : logit*factor
    // factor == 1.0 for non-penalized tokens -> identity operation
    vec_out_t result;
    result.x = l.x > 0.0f ? l.x / f.x : l.x * f.x;
    result.y = l.y > 0.0f ? l.y / f.y : l.y * f.y;
    result.z = l.z > 0.0f ? l.z / f.z : l.z * f.z;
    result.w = l.w > 0.0f ? l.w / f.w : l.w * f.w;
    dataOut[index] = result;
}
