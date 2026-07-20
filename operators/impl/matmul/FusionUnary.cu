// Native CUDA counterpart of FusionUnary.shader (matmul unary fusion module).
// The shader is a SPIR-V linkage module: it exports postProcessImpl (linked into a base
// matmul shader on the Vulkan path) and has no storage buffers / push constant, only a
// dummy [numthreads(1,1,1)] main() that keeps the helper alive — the shape the
// HLSL->CUDA transpiler rejects ("unsupported pattern"). On CUDA the equivalent unary
// fusion is inlined by MatMulCommon.cuh's mmWriteOutput, so this module is never
// dispatched standalone; it exists as the shader variant's native neighbor. dVal is
// ignored (unary fusion uses only the accumulated matmul result); cut_main is a dummy
// keep-alive matching the shader's [numthreads(1,1,1)] main().
#include "ComputeOpsShared.h"

// Fusion op selector — spec constant 2 (matches [[vk::constant_id(2)]] FUSION_OP).
#ifndef CUT_SPEC_2
#define CUT_SPEC_2 (0)
#endif
static const uint FUSION_OP = CUT_SPEC_2;

__device__ __noinline__ float postProcessImpl(float accum, float dVal) {
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

extern "C" __global__ void cut_main() {
    __shared__ float _keep;
    _keep = postProcessImpl(0.0f, 0.0f);
    (void)_keep;
}
