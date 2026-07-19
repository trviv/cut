// MatMulCommon.cuh
#pragma once

#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

// This header is the CUDA counterpart of MatMulCommon.shaderh (HLSL).
// Per-variant dtype macros (CUT_VEC_DTYPE_<SLOT>, CUT_SCALAR_DTYPE_<SLOT>,
// CUT_DTYPE_SIZE_<SLOT>, CUT_DTYPE_<SLOT>_IS_*) are passed by NVRTC -D defines
// from the native manifest. Every macro has an #ifndef default so the file
// reads standalone. Fusion is controlled by spec constants CUT_SPEC_1
// (FUSION_TYPE) and CUT_SPEC_2 (FUSION_OP).

#ifndef CUT_VEC_DTYPE_INPUT1
#define CUT_VEC_DTYPE_INPUT1 float4
#endif
#ifndef CUT_SCALAR_DTYPE_INPUT1
#define CUT_SCALAR_DTYPE_INPUT1 float
#endif

#ifndef CUT_VEC_DTYPE_INPUT2
#define CUT_VEC_DTYPE_INPUT2 float4
#endif
#ifndef CUT_SCALAR_DTYPE_INPUT2
#define CUT_SCALAR_DTYPE_INPUT2 float
#endif

#ifndef CUT_VEC_DTYPE_OUTPUT
#define CUT_VEC_DTYPE_OUTPUT float4
#endif
#ifndef CUT_SCALAR_DTYPE_OUTPUT
#define CUT_SCALAR_DTYPE_OUTPUT float
#endif

typedef CUT_VEC_DTYPE_INPUT1 cut_a_vec;   typedef CUT_SCALAR_DTYPE_INPUT1 cut_a_t;
typedef CUT_VEC_DTYPE_INPUT2 cut_b_vec;   typedef CUT_SCALAR_DTYPE_INPUT2 cut_b_t;
typedef CUT_VEC_DTYPE_OUTPUT cut_d_vec;   typedef CUT_SCALAR_DTYPE_OUTPUT cut_c_t;

struct PushConstants { uint M; uint K; uint N; uint strideA; uint strideB; uint strideC; };

#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (0)
#endif
static const uint FUSION_TYPE = CUT_SPEC_1;

#ifndef CUT_SPEC_2
#define CUT_SPEC_2 (0)
#endif
static const uint FUSION_OP = CUT_SPEC_2;

__device__ __forceinline__ half WaveReadLaneAt(half v, unsigned int lane) {
    return __ushort_as_half(__shfl_sync(0xffffffffu, __half_as_ushort(v), lane));
}

__device__ __forceinline__ cut_a_t mmLoadA(const cut_a_vec* __restrict__ dataA, const PushConstants& pc, uint row, uint col) {
    if (row >= pc.M || col >= pc.K) return (cut_a_t)(0);
    uint idx = row * pc.strideA + col;
    return dataA[idx >> 2][idx & 3];
}

__device__ __forceinline__ cut_a_t mmLoadAFast(const cut_a_vec* __restrict__ dataA, const PushConstants& pc, uint row, uint col) {
    uint idx = row * pc.strideA + col;
    return dataA[idx >> 2][idx & 3];
}

__device__ __forceinline__ cut_b_t mmLoadB(const cut_b_vec* __restrict__ dataB, const PushConstants& pc, uint row, uint col) {
    if (row >= pc.K || col >= pc.N) return (cut_b_t)(0);
    uint idx = row * pc.strideB + col;
    return dataB[idx >> 2][idx & 3];
}

__device__ __forceinline__ cut_b_t mmLoadBFast(const cut_b_vec* __restrict__ dataB, const PushConstants& pc, uint row, uint col) {
    uint idx = row * pc.strideB + col;
    return dataB[idx >> 2][idx & 3];
}

__device__ __forceinline__ uint2 mmSwizzle(const PushConstants& pc, uint spanM, uint spanN, uint gidx, uint gidy) {
    uint tilesN = (pc.N + spanN - 1) / spanN;
    uint tilesM = (pc.M + spanM - 1) / spanM;
    if (tilesM <= 1) return make_uint2(gidx, gidy);
    uint linearId = gidy * tilesN + gidx;
    return make_uint2(linearId / tilesM, linearId % tilesM);
}

__device__ __forceinline__ void mmWriteOutput(cut_c_t* __restrict__ dataC, const cut_d_vec* __restrict__ dataD, const PushConstants& pc, uint row, uint col, cut_c_t value) {
    uint idx = row * pc.strideC + col;
    if (FUSION_TYPE == 1) {
        float x = float(value);
        switch (FUSION_OP) {
            case OP_UNARY_NEG:        x = -x; break;
            case OP_UNARY_ABS:        x = abs(x); break;
            case OP_UNARY_SQRT:       x = sqrt(x); break;
            case OP_UNARY_SQUARE:     x = x * x; break;
            case OP_UNARY_RECIPROCAL: x = 1.0 / x; break;
            case OP_UNARY_EXP:        x = exp(x); break;
            case OP_UNARY_LOG:        x = log(x); break;
            case OP_UNARY_TANH:       x = tanh(x); break;
            case OP_UNARY_FLOOR:      x = floor(x); break;
            case OP_UNARY_CEIL:       x = ceil(x); break;
            case OP_UNARY_ROUND:      x = round(x); break;
            case OP_UNARY_RELU:       x = max(0.0, x); break;
            case OP_UNARY_SIGMOID:    x = 1.0 / (1.0 + exp(-x)); break;
            case OP_UNARY_GELU:       x = x * 0.5 * (1.0 + tanh(sqrt(2.0 / 3.14159265) * (x + 0.044715 * x * x * x))); break;
            case OP_UNARY_SILU:       x = x / (1.0 + exp(-x)); break;
            case OP_UNARY_SOFTPLUS:   x = log(1.0 + exp(x)); break;
            case OP_UNARY_RELU6:      x = clamp(x, 0.0, 6.0); break;
            case OP_UNARY_MISH:       x = x * tanh(log(1.0 + exp(x))); break;
            case OP_UNARY_HARDSWISH:  x = x * clamp(x / 6.0 + 0.5, 0.0, 1.0); break;
            case OP_UNARY_HARDSIGMOID: x = clamp(x / 6.0 + 0.5, 0.0, 1.0); break;
            case OP_UNARY_RSQRT:      x = rsqrt(x); break;
            default: break;
        }
        value = (cut_c_t)x;
    } else if (FUSION_TYPE == 2) {
        float a = float(value);
        float b = float(dataD[idx >> 2][idx & 3]);
        float r = a;
        switch (FUSION_OP) {
            case OP_BINARY_ADD:       r = a + b; break;
            case OP_BINARY_SUB:       r = a - b; break;
            case OP_BINARY_MUL:       r = a * b; break;
            case OP_BINARY_DIV:       r = a / b; break;
            case OP_BINARY_MOD:       r = a - b * floor(a / b); break;
            case OP_BINARY_POW:       r = pow(a, b); break;
            case OP_BINARY_FLOOR_DIV: r = floor(a / b); break;
            case OP_BINARY_MIN:       r = min(a, b); break;
            case OP_BINARY_MAX:       r = max(a, b); break;
            default:                  r = a + b; break;
        }
        value = (cut_c_t)r;
    }
    dataC[idx] = value;
}
