// Native CUDA counterpart of FusionBinaryOp.shader (matmul binary-op fusion module).
// The shader is a SPIR-V linkage module: it exports postProcessImpl (linked into a
// base matmul shader on the Vulkan path) and has no storage buffers / push constant,
// only a dummy [numthreads(1,1,1)] main() that keeps the helper alive — the shape the
// HLSL->CUDA transpiler rejects ("unsupported pattern"). On CUDA the equivalent binary
// fusion is inlined by MatMulCommon.cuh's mmWriteOutput, so this module is never
// dispatched standalone; it exists as the shader variant's native neighbor. cut_main is
// a dummy keep-alive matching the shader's [numthreads(1,1,1)] main().
#include "ComputeOpsShared.h"

// Fusion op selector — spec constant 2 (matches [[vk::constant_id(2)]] FUSION_OP).
#ifndef CUT_SPEC_2
#define CUT_SPEC_2 (0)
#endif
static const uint FUSION_OP = CUT_SPEC_2;

__device__ __noinline__ float postProcessImpl(float accum, float dVal) {
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

extern "C" __global__ void cut_main() {
    __shared__ float _keep;
    _keep = postProcessImpl(0.0f, 0.0f);
    (void)_keep;
}
