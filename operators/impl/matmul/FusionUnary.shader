#include "ComputeOpsShared.h"

// Unary fusion module for compileCustomShader.
// postProcessImpl replaces the postProcess stub in base matmul shaders.
// Specialization constant selects the unary operation applied to accum.
// dVal is ignored — unary fusion only uses the accumulated matmul result.

[[vk::constant_id(2)]] const uint FUSION_OP = 0;

[noinline] float postProcessImpl(float accum, float dVal) {
    float x = accum;
    switch (FUSION_OP) {
        case OP_UNARY_NEG:        return -x;
        case OP_UNARY_ABS:        return abs(x);
        case OP_UNARY_SQRT:       return sqrt(x);
        case OP_UNARY_SQUARE:     return x * x;
        case OP_UNARY_RECIPROCAL: return 1.0 / x;
        case OP_UNARY_EXP:        return exp(x);
        case OP_UNARY_LOG:        return log(x);
        case OP_UNARY_TANH:       return tanh(x);
        case OP_UNARY_FLOOR:      return floor(x);
        case OP_UNARY_CEIL:       return ceil(x);
        case OP_UNARY_ROUND:      return round(x);
        case OP_UNARY_RELU:       return max(0.0, x);
        case OP_UNARY_SIGMOID:    return 1.0 / (1.0 + exp(-x));
        case OP_UNARY_GELU:       return x * 0.5 * (1.0 + tanh(sqrt(2.0 / 3.14159265) * (x + 0.044715 * x * x * x)));
        case OP_UNARY_SILU:       return x / (1.0 + exp(-x));
        case OP_UNARY_SOFTPLUS:   return log(1.0 + exp(x));
        case OP_UNARY_RELU6:      return clamp(x, 0.0, 6.0);
        case OP_UNARY_MISH:       return x * tanh(log(1.0 + exp(x)));
        case OP_UNARY_HARDSWISH:  return x * clamp(x / 6.0 + 0.5, 0.0, 1.0);
        case OP_UNARY_HARDSIGMOID: return clamp(x / 6.0 + 0.5, 0.0, 1.0);
        case OP_UNARY_RSQRT:      return rsqrt(x);
        default:                  return x;
    }
}

groupshared float _keep;

[numthreads(1, 1, 1)]
void main() { _keep = postProcessImpl(0, 0); }
