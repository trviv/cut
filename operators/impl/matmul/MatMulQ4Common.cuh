#pragma once
#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

// dtype macro defaults (standalone readable). INPUT1 and OUTPUT default float; SCALES default half.
#ifndef CUT_VEC_DTYPE_INPUT1
#define CUT_VEC_DTYPE_INPUT1 float4
#endif
#ifndef CUT_SCALAR_DTYPE_INPUT1
#define CUT_SCALAR_DTYPE_INPUT1 float
#endif
#ifndef CUT_VEC_DTYPE_SCALES
#define CUT_VEC_DTYPE_SCALES half4
#endif
#ifndef CUT_VEC_DTYPE_OUTPUT
#define CUT_VEC_DTYPE_OUTPUT float4
#endif
#ifndef CUT_SCALAR_DTYPE_OUTPUT
#define CUT_SCALAR_DTYPE_OUTPUT float
#endif

struct PushConstants {
    uint M;
    uint K;
    uint N;
    uint strideA;
    uint strideBNpacked;
    uint strideC;
    uint scaleStride;
};

// Fusion spec constants (runtime supplies CUT_SPEC_1 / CUT_SPEC_2)
#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (0)
#endif
static const uint FUSION_TYPE = CUT_SPEC_1;
#ifndef CUT_SPEC_2
#define CUT_SPEC_2 (0)
#endif
static const uint FUSION_OP = CUT_SPEC_2;

// swizzleTileId only when tiling defines present
#if defined(TILE_SIZE) && defined(TM) && defined(TN)
__device__ __forceinline__ uint2 cut_swizzleTileId(const PushConstants& pc, uint3 Gid) {
    uint tilesN = (pc.N + TILE_SIZE * TN - 1) / (TILE_SIZE * TN);
    uint tilesM = (pc.M + TILE_SIZE * TM - 1) / (TILE_SIZE * TM);
    if (tilesM <= 1) return make_uint2(Gid.x, Gid.y);
    uint linearId = Gid.y * tilesN + Gid.x;
    return make_uint2(linearId / tilesM, linearId % tilesM);
}
#endif

// writeOutput: same fusion switch as the .shaderh, value is CUT_SCALAR_DTYPE_OUTPUT (float).
// dataD element is CUT_VEC_DTYPE_OUTPUT (float4), read as dataD[idx>>2][idx&3].
__device__ __forceinline__ void cut_writeOutput(CUT_SCALAR_DTYPE_OUTPUT* __restrict__ dataC,
                                                const CUT_VEC_DTYPE_OUTPUT* __restrict__ dataD,
                                                const PushConstants& pc,
                                                uint row, uint col, CUT_SCALAR_DTYPE_OUTPUT value) {
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
        value = (CUT_SCALAR_DTYPE_OUTPUT)x;
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
        value = (CUT_SCALAR_DTYPE_OUTPUT)r;
    }
    dataC[idx] = value;
}

// loadA: bounds-checked, returns CUT_SCALAR_DTYPE_INPUT1
__device__ __forceinline__ CUT_SCALAR_DTYPE_INPUT1 cut_loadA(const CUT_VEC_DTYPE_INPUT1* __restrict__ dataA,
                                                             const PushConstants& pc, uint row, uint col) {
    if (row >= pc.M || col >= pc.K) return (CUT_SCALAR_DTYPE_INPUT1)(0);
    uint idx = row * pc.strideA + col;
    return dataA[idx >> 2][idx & 3];
}

// loadB: Q4_0 nibble dequant, bounds-checked. COPY logic from transpiled reference loadB EXACTLY
// (halfN, byteIdx uses pc.strideBNpacked, extract nibble, val = int(nibble)-8, scale from scalesB).
__device__ __forceinline__ float cut_loadB(const uint* __restrict__ packedB,
                                            const CUT_VEC_DTYPE_SCALES* __restrict__ scalesB,
                                            const PushConstants& pc, uint k, uint n) {
    if (k >= pc.K || n >= pc.N) return 0.0f;

    // Locate the byte holding this nibble in packed B [K, N/2]
    uint halfN = n >> 1;
    uint byteIdx = k * pc.strideBNpacked + halfN;
    uint packed = packedB[byteIdx >> 2];
    uint byteShift = (byteIdx & 3u) * 8u;
    uint byte_val = (packed >> byteShift) & 0xFFu;

    // Extract the correct nibble: lower for even n, upper for odd n
    uint nibbleShift = (n & 1u) * 4u;
    int val = int((byte_val >> nibbleShift) & 0xFu) - 8;

    // Read per-block scale from scalesB[k/32][n]
    uint scaleIdx = (k >> 5) * pc.scaleStride + n;
    float scale = float(scalesB[scaleIdx >> 2][scaleIdx & 3]);

    return float(val) * scale;
}

// Shared helper: load 4 consecutive column scales starting at scaleBase (half path only for Q4).
__device__ __forceinline__ void cut_loadScale4(const CUT_VEC_DTYPE_SCALES* __restrict__ scalesB,
                                                uint scaleBase,
                                                float& s0, float& s1, float& s2, float& s3) {
    CUT_VEC_DTYPE_SCALES sv = scalesB[scaleBase >> 2];
    s0 = float(sv[0]); s1 = float(sv[1]); s2 = float(sv[2]); s3 = float(sv[3]);
}
