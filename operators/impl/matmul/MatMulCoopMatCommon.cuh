#pragma once

// Shared device helpers for the native CUDA cooperative-matrix matmul kernels.
//
// Tensor-core MMA via INLINE PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
// (Ampere HMMA), NOT nvcuda::wmma: the runtime NVRTC include dir provides mma.h without its
// crt/mma.h dependency, so <mma.h> is unusable here. Inline PTX needs only cuda_fp16.h.
//
// A 16x16x16 fp16->fp32 tile multiply is built from two m16n8k16 instructions (N split 8+8).
// Fragment register<->element layout follows the PTX ISA (mma.m16n8k16, .f16). Per warp lane:
//   gid = lane >> 2 (0..7);  t = (lane & 3) * 2 (0,2,4,6).

#include "cut_cuda_prelude.cuh"
#include "ComputeOpsShared.h"

// One 16x16 A tile (row-major), held as 4 x b32 (2 fp16 each) per lane.
struct FragA16 { unsigned int r[4]; };
// One 16x16 B tile (row-major in memory), held as the col-major m16n8k16 B fragment for each
// N-half: lo = N cols 0..7, hi = N cols 8..15. Each half is 2 x b32 per lane.
struct FragB16 { unsigned int lo[2]; unsigned int hi[2]; };
// One 16x16 fp32 accumulator, split into the two N-halves (4 f32 each per lane).
struct Acc16 { float lo[4]; float hi[4]; };

__device__ __forceinline__ void zeroAcc16(Acc16 &c) {
    #pragma unroll
    for (int i = 0; i < 4; i++) { c.lo[i] = 0.0f; c.hi[i] = 0.0f; }
}

// Pack two fp16 values into a b32 register: low 16 bits = a, high 16 bits = b.
__device__ __forceinline__ unsigned int cutPack2(half a, half b) {
    __half2 h = __halves2half2(a, b);
    return *reinterpret_cast<unsigned int *>(&h);
}

// Load a 16x16 A tile (row-major) from `s` with leading dimension `ldm` (in fp16 elements).
// The two adjacent columns of each b32 are contiguous in memory, so read them as one aligned
// b32. All call sites use even ldm (16/32) and even base offsets; buffers/shared are >=4-byte
// aligned, so the b32 reads are aligned.
__device__ __forceinline__ void cutLoadA16(FragA16 &a, const half *s, int ldm, int lane) {
    int gid = lane >> 2;
    int t = (lane & 3) * 2;
    a.r[0] = *reinterpret_cast<const unsigned int *>(s + (gid    ) * ldm + t);
    a.r[1] = *reinterpret_cast<const unsigned int *>(s + (gid + 8) * ldm + t);
    a.r[2] = *reinterpret_cast<const unsigned int *>(s + (gid    ) * ldm + t + 8);
    a.r[3] = *reinterpret_cast<const unsigned int *>(s + (gid + 8) * ldm + t + 8);
}

// Load a 16x16 B tile (row-major in memory, element B[k][n] = s[k*ldm + n]) into the col-major
// m16n8k16 B fragment for both N-halves. The two fp16 in each b32 are K-adjacent (ldm apart in
// memory), so they are packed explicitly rather than read as one word.
__device__ __forceinline__ void cutLoadB16(FragB16 &b, const half *s, int ldm, int lane) {
    int gid = lane >> 2;
    int t = (lane & 3) * 2;
    b.lo[0] = cutPack2(s[(t    ) * ldm + gid    ], s[(t + 1) * ldm + gid    ]);
    b.lo[1] = cutPack2(s[(t + 8) * ldm + gid    ], s[(t + 9) * ldm + gid    ]);
    b.hi[0] = cutPack2(s[(t    ) * ldm + gid + 8], s[(t + 1) * ldm + gid + 8]);
    b.hi[1] = cutPack2(s[(t + 8) * ldm + gid + 8], s[(t + 9) * ldm + gid + 8]);
}

// One m16n8k16 HMMA, accumulating in place: d[0..3] += A(a[0..3]) * B(b[0..1]).
__device__ __forceinline__ void cutMmaN8(float d[4], const unsigned int a[4],
                                          const unsigned int b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 16x16x16 accumulate: c += a * b (issues the two N-half m16n8k16 instructions).
__device__ __forceinline__ void cutMma16(Acc16 &c, const FragA16 &a, const FragB16 &b) {
    cutMmaN8(c.lo, a.r, b.lo);
    cutMmaN8(c.hi, a.r, b.hi);
}

// Store a 16x16 fp32 accumulator to `out` (row-major, leading dimension `ldm`). `out` may point
// at global or shared memory; the full 16x16 tile is written (caller does any tile-level bounds
// checking, exactly as the .comp does before coopMatStore).
__device__ __forceinline__ void cutStoreC16(float *out, int ldm, const Acc16 &c, int lane) {
    int gid = lane >> 2;
    int t = (lane & 3) * 2;
    out[(gid    ) * ldm + t + 0]     = c.lo[0];
    out[(gid    ) * ldm + t + 1]     = c.lo[1];
    out[(gid + 8) * ldm + t + 0]     = c.lo[2];
    out[(gid + 8) * ldm + t + 1]     = c.lo[3];
    out[(gid    ) * ldm + 8 + t + 0] = c.hi[0];
    out[(gid    ) * ldm + 8 + t + 1] = c.hi[1];
    out[(gid + 8) * ldm + 8 + t + 0] = c.hi[2];
    out[(gid + 8) * ldm + 8 + t + 1] = c.hi[3];
}

// Push constants for the fp16 coopmat kernels (matches the GLSL push_constant block, all uint, tightly packed).
struct CoopMatPush { uint M; uint K; uint N; uint strideA; uint strideB; uint strideC; };
// Push constants for the Q8 coopmat kernel (adds byte-stride for int8 B and a scale stride).
struct Q8CoopMatPush { uint M; uint K; uint N; uint strideA; uint strideBN; uint strideC; uint scaleStride; };
// Push constants for the Q4 coopmat kernel: B is packed two nibbles per byte,
// so its row stride is in packed bytes.
struct Q4CoopMatPush { uint M; uint K; uint N; uint strideA; uint strideBNpacked; uint strideC; uint scaleStride; };

// Fusion spec constants (runtime supplies CUT_SPEC_1 / CUT_SPEC_2, mirroring
// the FUSION_TYPE / FUSION_OP specialization constants in the .comp).
#ifndef CUT_SPEC_1
#define CUT_SPEC_1 (0)
#endif
static const uint COOPMAT_FUSION_TYPE = CUT_SPEC_1;
#ifndef CUT_SPEC_2
#define CUT_SPEC_2 (0)
#endif
static const uint COOPMAT_FUSION_OP = CUT_SPEC_2;

// Affine (zero-point) Q4: when 1, each 32-weight block carries a min in the
// upper half of the scales buffer. Mirrors AFFINE in MatMulQ4Common.
#ifndef CUT_SPEC_3
#define CUT_SPEC_3 (0)
#endif
static const uint COOPMAT_AFFINE = CUT_SPEC_3;

// Applies the fused epilogue to one output element. `d` is the corresponding
// dataD element (only read when a binary fusion is active).
__device__ __forceinline__ float cutApplyFusion(float value, float d) {
    if (COOPMAT_FUSION_TYPE == 1u) {
        float x = value;
        switch (COOPMAT_FUSION_OP) {
            case OP_UNARY_NEG:        x = -x; break;
            case OP_UNARY_ABS:        x = abs(x); break;
            case OP_UNARY_SQRT:       x = sqrt(x); break;
            case OP_UNARY_SQUARE:     x = x * x; break;
            case OP_UNARY_RECIPROCAL: x = 1.0f / x; break;
            case OP_UNARY_EXP:        x = exp(x); break;
            case OP_UNARY_LOG:        x = log(x); break;
            case OP_UNARY_TANH:       x = tanh(x); break;
            case OP_UNARY_FLOOR:      x = floor(x); break;
            case OP_UNARY_CEIL:       x = ceil(x); break;
            case OP_UNARY_ROUND:      x = round(x); break;
            case OP_UNARY_RELU:       x = max(0.0f, x); break;
            case OP_UNARY_SIGMOID:    x = 1.0f / (1.0f + exp(-x)); break;
            case OP_UNARY_GELU:       x = x * 0.5f * (1.0f + tanh(sqrt(2.0f / 3.14159265f) * (x + 0.044715f * x * x * x))); break;
            case OP_UNARY_SILU:       x = x / (1.0f + exp(-x)); break;
            case OP_UNARY_SOFTPLUS:   x = log(1.0f + exp(x)); break;
            case OP_UNARY_RELU6:      x = fminf(fmaxf(x, 0.0f), 6.0f); break;
            case OP_UNARY_MISH:       x = x * tanh(log(1.0f + exp(x))); break;
            case OP_UNARY_HARDSWISH:  x = x * fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f); break;
            case OP_UNARY_HARDSIGMOID: x = fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f); break;
            case OP_UNARY_RSQRT:      x = rsqrt(x); break;
            default: break;
        }
        return x;
    } else if (COOPMAT_FUSION_TYPE == 2u) {
        float a = value;
        float b = d;
        float r = a;
        switch (COOPMAT_FUSION_OP) {
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
        return r;
    }
    return value;
}
