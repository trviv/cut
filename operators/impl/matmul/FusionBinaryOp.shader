#include "ComputeOpsShared.h"

// Binary op fusion module for compileCustomShader.
// postProcessImpl replaces the postProcess stub in base matmul shaders.
// Specialization constant selects the binary operation applied to (accum, dVal).

[[vk::constant_id(2)]] const uint FUSION_OP = 0;

[noinline] float postProcessImpl(float accum, float dVal) {
    switch (FUSION_OP) {
        case OP_BINARY_ADD:       return accum + dVal;
        case OP_BINARY_SUB:       return accum - dVal;
        case OP_BINARY_MUL:       return accum * dVal;
        case OP_BINARY_DIV:       return accum / dVal;
        case OP_BINARY_MOD:       return (accum - dVal * floor(accum / dVal));
        case OP_BINARY_POW:       return pow(accum, dVal);
        case OP_BINARY_FLOOR_DIV: return floor(accum / dVal);
        case OP_BINARY_MIN:       return min(accum, dVal);
        case OP_BINARY_MAX:       return max(accum, dVal);
        default:                  return accum + dVal;
    }
}

groupshared float _keep;

[numthreads(1, 1, 1)]
void main() { _keep = postProcessImpl(0, 0); }
