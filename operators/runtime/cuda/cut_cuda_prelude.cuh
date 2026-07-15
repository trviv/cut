// CUDA shim prelude for HLSL-transpiled compute kernels.
//
// The CUT operators are authored in HLSL (compiled to SPIR-V for Vulkan). For
// the CUDA backend the same HLSL bodies are transpiled (mostly textually) into
// CUDA C++ and compiled at runtime with NVRTC. This header provides the shims
// that let the HLSL bodies compile and behave correctly as CUDA:
//
//   * vector types (float4/int4/uint4 are CUDA builtins; half4 is defined here)
//   * arithmetic operators for those vector types (CUDA builtins have none)
//   * HLSL intrinsic overloads operating element-wise on vectors
//   * cut_cast_* helpers that replace HLSL's "(float4)(x)" broadcast casts
//
// Only the element-wise (unary/binary/ternary/cast) families are covered today;
// kernels using shared memory, group ops, or matrix tiles need more shims.
#pragma once

typedef unsigned int uint;

// cuda_fp16.h is included BEFORE we shadow the vector type names below, so its
// internal use of the builtin vector types is unaffected.
#include <cuda_fp16.h>

// ===========================================================================
// Vector types. HLSL float4/int4/uint4 support subscripting (v[i]); the builtin
// CUDA vector types do not. We define layout-compatible structs (4 contiguous
// lanes, same size/alignment) that add operator[], and shadow the builtin names
// so transpiled kernels and the shims below use them uniformly.
// ===========================================================================
#define CUT_VEC4(NAME, T)                                                      \
  struct NAME {                                                                \
    T x, y, z, w;                                                              \
    __device__ __forceinline__ T &operator[](int i) { return (&x)[i]; }        \
    __device__ __forceinline__ const T &operator[](int i) const {              \
      return (&x)[i];                                                          \
    }                                                                          \
  };
CUT_VEC4(cut_vec4f, float)
CUT_VEC4(cut_vec4i, int)
CUT_VEC4(cut_vec4u, unsigned int)
CUT_VEC4(half4, half)
#undef CUT_VEC4
#define float4 cut_vec4f
#define int4 cut_vec4i
#define uint4 cut_vec4u

// ===========================================================================
// Scalar HLSL intrinsics that CUDA device code lacks (used by tiled/indexed
// kernels operating on scalars rather than vec4 lanes).
// ===========================================================================
// Scalar clamp. The (float,double,double) overload covers the common
// clamp(x, 0.0, 6.0) with double literals; explicit types avoid ambiguity.
__device__ __forceinline__ float clamp(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}
__device__ __forceinline__ float clamp(float x, double lo, double hi) {
  return fminf(fmaxf(x, (float)lo), (float)hi);
}
__device__ __forceinline__ int clamp(int x, int lo, int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
__device__ __forceinline__ unsigned int clamp(unsigned int x, unsigned int lo,
                                               unsigned int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
__device__ __forceinline__ float saturate(float x) {
  return fminf(fmaxf(x, 0.0f), 1.0f);
}
__device__ __forceinline__ float lerp(float a, float b, float t) {
  return a + t * (b - a);
}
__device__ __forceinline__ float frac(float x) { return x - floorf(x); }
__device__ __forceinline__ float rsqrt(float x) { return rsqrtf(x); }
__device__ __forceinline__ float sign(float x) {
  return (float)((x > 0.0f) - (x < 0.0f));
}
// mad(a,b,c) = a*b+c (HLSL multiply-add).
__device__ __forceinline__ float mad(float a, float b, float c) {
  return fmaf(a, b, c);
}
__device__ __forceinline__ int mad(int a, int b, int c) { return a * b + c; }
__device__ __forceinline__ unsigned int mad(unsigned int a, unsigned int b,
                                             unsigned int c) {
  return a * b + c;
}
// half scalar intrinsics (half-precision tiled kernels accumulate in half).
__device__ __forceinline__ half mad(half a, half b, half c) {
  return a * b + c;
}
__device__ __forceinline__ half min(half a, half b) { return a < b ? a : b; }
__device__ __forceinline__ half max(half a, half b) { return a > b ? a : b; }

// WaveReadLaneAt(value, lane) -> warp shuffle (HLSL wave == CUDA warp, 32-wide).
__device__ __forceinline__ float WaveReadLaneAt(float v, unsigned int lane) {
  return __shfl_sync(0xffffffffu, v, lane);
}
__device__ __forceinline__ int WaveReadLaneAt(int v, unsigned int lane) {
  return __shfl_sync(0xffffffffu, v, lane);
}
__device__ __forceinline__ unsigned int WaveReadLaneAt(unsigned int v,
                                                       unsigned int lane) {
  return __shfl_sync(0xffffffffu, v, lane);
}

// ===========================================================================
// Constructors / broadcast casts (replace HLSL "(T)(expr)").
// ===========================================================================
__device__ __forceinline__ float4 cut_mk_f4(float x, float y, float z, float w) {
  float4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r;
}
__device__ __forceinline__ int4 cut_mk_i4(int x, int y, int z, int w) {
  int4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r;
}
__device__ __forceinline__ uint4 cut_mk_u4(uint x, uint y, uint z, uint w) {
  uint4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r;
}

// Single-argument constructor forms (HLSL "float4(expr)"): identity for a
// same-type vector and a scalar splat. The half4-widening form lives below,
// after cut_h2f.
__device__ __forceinline__ float4 cut_mk_f4(float4 v) { return v; }
__device__ __forceinline__ float4 cut_mk_f4(float s) {
  return cut_mk_f4(s, s, s, s);
}

// cut_cast_f4 — broadcast a scalar, identity for float4, convert int4 masks/ints.
__device__ __forceinline__ float4 cut_cast_f4(float s) { return cut_mk_f4(s, s, s, s); }
__device__ __forceinline__ float4 cut_cast_f4(double s) { return cut_cast_f4((float)s); }
__device__ __forceinline__ float4 cut_cast_f4(int s) { return cut_cast_f4((float)s); }
__device__ __forceinline__ float4 cut_cast_f4(uint s) { return cut_cast_f4((float)s); }
__device__ __forceinline__ float4 cut_cast_f4(float4 v) { return v; }
__device__ __forceinline__ float4 cut_cast_f4(int4 v) {
  return cut_mk_f4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}
__device__ __forceinline__ float4 cut_cast_f4(uint4 v) {
  return cut_mk_f4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}

__device__ __forceinline__ int4 cut_cast_i4(int s) { return cut_mk_i4(s, s, s, s); }
__device__ __forceinline__ int4 cut_cast_i4(float s) { return cut_cast_i4((int)s); }
__device__ __forceinline__ int4 cut_cast_i4(int4 v) { return v; }
__device__ __forceinline__ int4 cut_cast_i4(float4 v) {
  return cut_mk_i4((int)v.x, (int)v.y, (int)v.z, (int)v.w);
}

__device__ __forceinline__ uint4 cut_cast_u4(uint s) { return cut_mk_u4(s, s, s, s); }
__device__ __forceinline__ uint4 cut_cast_u4(int s) { return cut_cast_u4((uint)s); }
__device__ __forceinline__ uint4 cut_cast_u4(uint4 v) { return v; }
__device__ __forceinline__ uint4 cut_cast_u4(int4 v) {
  return cut_mk_u4((uint)v.x, (uint)v.y, (uint)v.z, (uint)v.w);
}

__device__ __forceinline__ half4 cut_cast_h4(float s) {
  half4 r; half h = __float2half(s); r.x = h; r.y = h; r.z = h; r.w = h; return r;
}
__device__ __forceinline__ half4 cut_cast_h4(double s) { return cut_cast_h4((float)s); }
__device__ __forceinline__ half4 cut_cast_h4(half s) {
  half4 r; r.x = s; r.y = s; r.z = s; r.w = s; return r;
}
__device__ __forceinline__ half4 cut_cast_h4(int s) { return cut_cast_h4((float)s); }
__device__ __forceinline__ half4 cut_cast_h4(uint s) { return cut_cast_h4((float)s); }
__device__ __forceinline__ half4 cut_cast_h4(half4 v) { return v; }

// ===========================================================================
// float4 arithmetic operators (scalar promotes via cut_cast_f4).
// ===========================================================================
#define CUT_F4_BINOP(op)                                                       \
  __device__ __forceinline__ float4 operator op(float4 a, float4 b) {          \
    return cut_mk_f4(a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w);           \
  }                                                                            \
  __device__ __forceinline__ float4 operator op(float4 a, float b) {           \
    return a op cut_cast_f4(b);                                                 \
  }                                                                            \
  __device__ __forceinline__ float4 operator op(float a, float4 b) {           \
    return cut_cast_f4(a) op b;                                                 \
  }                                                                            \
  __device__ __forceinline__ float4 operator op(float4 a, double b) {          \
    return a op cut_cast_f4(b);                                                 \
  }                                                                            \
  __device__ __forceinline__ float4 operator op(double a, float4 b) {          \
    return cut_cast_f4(a) op b;                                                 \
  }
CUT_F4_BINOP(+)
CUT_F4_BINOP(-)
CUT_F4_BINOP(*)
CUT_F4_BINOP(/)
#undef CUT_F4_BINOP

// Compound assignments (HLSL "a += b;" etc. on float4).
#define CUT_F4_COMPOUND(op)                                                    \
  __device__ __forceinline__ float4 &operator op##=(float4 &a, float4 b) {     \
    a = a op b;                                                                \
    return a;                                                                  \
  }                                                                            \
  __device__ __forceinline__ float4 &operator op##=(float4 &a, float b) {      \
    a = a op cut_cast_f4(b);                                                   \
    return a;                                                                  \
  }
CUT_F4_COMPOUND(+)
CUT_F4_COMPOUND(-)
CUT_F4_COMPOUND(*)
CUT_F4_COMPOUND(/)
#undef CUT_F4_COMPOUND

__device__ __forceinline__ float4 operator-(float4 a) {
  return cut_mk_f4(-a.x, -a.y, -a.z, -a.w);
}

// Comparisons return an int4 mask (1/0), matching HLSL vector compares which
// then get cast back to a numeric vector via (float4)(...).
#define CUT_F4_CMP(op)                                                         \
  __device__ __forceinline__ int4 operator op(float4 a, float4 b) {            \
    return cut_mk_i4(a.x op b.x ? 1 : 0, a.y op b.y ? 1 : 0,                    \
                     a.z op b.z ? 1 : 0, a.w op b.w ? 1 : 0);                   \
  }                                                                            \
  __device__ __forceinline__ int4 operator op(float4 a, float b) {             \
    return a op cut_cast_f4(b);                                                 \
  }
CUT_F4_CMP(==)
CUT_F4_CMP(!=)
CUT_F4_CMP(<)
CUT_F4_CMP(<=)
CUT_F4_CMP(>)
CUT_F4_CMP(>=)
#undef CUT_F4_CMP

// int4 bitwise / logical (for bitwise-not and mask negation).
__device__ __forceinline__ int4 operator~(int4 a) {
  return cut_mk_i4(~a.x, ~a.y, ~a.z, ~a.w);
}
__device__ __forceinline__ uint4 operator~(uint4 a) {
  return cut_mk_u4(~a.x, ~a.y, ~a.z, ~a.w);
}
__device__ __forceinline__ int4 operator!(int4 a) {
  return cut_mk_i4(!a.x, !a.y, !a.z, !a.w);
}

// ===========================================================================
// HLSL math intrinsics, element-wise over float4.
// ===========================================================================
#define CUT_F4_UNARY(name, fn)                                                 \
  __device__ __forceinline__ float4 name(float4 a) {                           \
    return cut_mk_f4(fn(a.x), fn(a.y), fn(a.z), fn(a.w));                       \
  }
CUT_F4_UNARY(abs, fabsf)
CUT_F4_UNARY(sqrt, sqrtf)
CUT_F4_UNARY(rsqrt, rsqrtf)
CUT_F4_UNARY(exp, expf)
CUT_F4_UNARY(exp2, exp2f)
CUT_F4_UNARY(log, logf)
CUT_F4_UNARY(log2, log2f)
CUT_F4_UNARY(sin, sinf)
CUT_F4_UNARY(cos, cosf)
CUT_F4_UNARY(tan, tanf)
CUT_F4_UNARY(asin, asinf)
CUT_F4_UNARY(acos, acosf)
CUT_F4_UNARY(atan, atanf)
CUT_F4_UNARY(sinh, sinhf)
CUT_F4_UNARY(cosh, coshf)
CUT_F4_UNARY(tanh, tanhf)
CUT_F4_UNARY(floor, floorf)
CUT_F4_UNARY(ceil, ceilf)
CUT_F4_UNARY(trunc, truncf)
#undef CUT_F4_UNARY

__device__ __forceinline__ float cut_signf(float x) {
  return (float)((x > 0.0f) - (x < 0.0f));
}
__device__ __forceinline__ float4 sign(float4 a) {
  return cut_mk_f4(cut_signf(a.x), cut_signf(a.y), cut_signf(a.z), cut_signf(a.w));
}
__device__ __forceinline__ float cut_fracf(float x) { return x - floorf(x); }
__device__ __forceinline__ float4 frac(float4 a) {
  return cut_mk_f4(cut_fracf(a.x), cut_fracf(a.y), cut_fracf(a.z), cut_fracf(a.w));
}
__device__ __forceinline__ float4 degrees(float4 a) {
  return a * (180.0f / 3.14159265358979323846f);
}
__device__ __forceinline__ float4 radians(float4 a) {
  return a * (3.14159265358979323846f / 180.0f);
}
__device__ __forceinline__ int4 isnan(float4 a) {
  return cut_mk_i4(isnan(a.x), isnan(a.y), isnan(a.z), isnan(a.w));
}
__device__ __forceinline__ int4 isinf(float4 a) {
  return cut_mk_i4(isinf(a.x), isinf(a.y), isinf(a.z), isinf(a.w));
}
__device__ __forceinline__ float4 pow(float4 a, float4 b) {
  return cut_mk_f4(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z), powf(a.w, b.w));
}

// min / max / clamp (scalar args promote to float4).
#define CUT_F4_MINMAX(name, fn)                                                \
  __device__ __forceinline__ float4 name(float4 a, float4 b) {                 \
    return cut_mk_f4(fn(a.x, b.x), fn(a.y, b.y), fn(a.z, b.z), fn(a.w, b.w));   \
  }                                                                            \
  __device__ __forceinline__ float4 name(float4 a, float b) {                  \
    return name(a, cut_cast_f4(b));                                            \
  }                                                                            \
  __device__ __forceinline__ float4 name(float a, float4 b) {                  \
    return name(cut_cast_f4(a), b);                                            \
  }
CUT_F4_MINMAX(min, fminf)
CUT_F4_MINMAX(max, fmaxf)
#undef CUT_F4_MINMAX

__device__ __forceinline__ float4 clamp(float4 a, float4 lo, float4 hi) {
  return min(max(a, lo), hi);
}
__device__ __forceinline__ float4 clamp(float4 a, float lo, float hi) {
  return min(max(a, cut_cast_f4(lo)), cut_cast_f4(hi));
}

// HLSL lerp(a, b, t) = a + t * (b - a).
__device__ __forceinline__ float4 lerp(float4 a, float4 b, float4 t) {
  return a + t * (b - a);
}

// Bit reinterpretation.
__device__ __forceinline__ float4 asfloat(int4 v) {
  return cut_mk_f4(__int_as_float(v.x), __int_as_float(v.y),
                   __int_as_float(v.z), __int_as_float(v.w));
}
__device__ __forceinline__ int4 asint(float4 v) {
  return cut_mk_i4(__float_as_int(v.x), __float_as_int(v.y),
                   __float_as_int(v.z), __float_as_int(v.w));
}

// Scalar bit reinterpretation (HLSL asfloat/asint/asuint on scalars).
__device__ __forceinline__ float asfloat(uint x) { return __uint_as_float(x); }
__device__ __forceinline__ float asfloat(int x) { return __int_as_float(x); }
__device__ __forceinline__ float asfloat(float x) { return x; }
__device__ __forceinline__ int asint(float x) { return __float_as_int(x); }
__device__ __forceinline__ uint asuint(float x) { return __float_as_uint(x); }

// ===========================================================================
// Binary-family additions: float modulo, atan2, and integer vector operators.
// ===========================================================================
__device__ __forceinline__ float4 operator%(float4 a, float4 b) {
  return cut_mk_f4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z),
                   fmodf(a.w, b.w));
}
__device__ __forceinline__ float4 operator%(float4 a, float b) {
  return a % cut_cast_f4(b);
}
__device__ __forceinline__ float4 atan2(float4 a, float4 b) {
  return cut_mk_f4(atan2f(a.x, b.x), atan2f(a.y, b.y), atan2f(a.z, b.z),
                   atan2f(a.w, b.w));
}
__device__ __forceinline__ float4 mad(float4 a, float4 b, float4 c) {
  return a * b + c;
}

// int4 / uint4 element-wise operators (bitwise paths + integer variants).
#define CUT_INT4_BINOP(T, MK, op)                                              \
  __device__ __forceinline__ T operator op(T a, T b) {                         \
    return MK(a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w);                  \
  }
#define CUT_INT4_ALL(T, MK)                                                    \
  CUT_INT4_BINOP(T, MK, +) CUT_INT4_BINOP(T, MK, -)                            \
  CUT_INT4_BINOP(T, MK, *) CUT_INT4_BINOP(T, MK, /)                            \
  CUT_INT4_BINOP(T, MK, %) CUT_INT4_BINOP(T, MK, &)                            \
  CUT_INT4_BINOP(T, MK, |) CUT_INT4_BINOP(T, MK, ^)
CUT_INT4_ALL(int4, cut_mk_i4)
CUT_INT4_ALL(uint4, cut_mk_u4)
#undef CUT_INT4_ALL
#undef CUT_INT4_BINOP

// Shifts mask the count to 5 bits to match SPIR-V/HLSL semantics (CUDA's <<,>>
// are undefined for counts >= bit width, so bit-reinterpret float shifts with
// large counts would otherwise produce 0 instead of the masked result).
#define CUT_INT4_SHIFT(T, MK)                                                  \
  __device__ __forceinline__ T operator<<(T a, T b) {                          \
    return MK(a.x << (b.x & 31), a.y << (b.y & 31), a.z << (b.z & 31),          \
             a.w << (b.w & 31));                                               \
  }                                                                            \
  __device__ __forceinline__ T operator>>(T a, T b) {                          \
    return MK(a.x >> (b.x & 31), a.y >> (b.y & 31), a.z >> (b.z & 31),          \
             a.w >> (b.w & 31));                                               \
  }
CUT_INT4_SHIFT(int4, cut_mk_i4)
CUT_INT4_SHIFT(uint4, cut_mk_u4)
#undef CUT_INT4_SHIFT

// int4 / uint4 comparisons return an int4 mask (matching HLSL vector compares).
#define CUT_INT4_CMP(T, op)                                                    \
  __device__ __forceinline__ int4 operator op(T a, T b) {                      \
    return cut_mk_i4(a.x op b.x ? 1 : 0, a.y op b.y ? 1 : 0,                    \
                     a.z op b.z ? 1 : 0, a.w op b.w ? 1 : 0);                   \
  }
#define CUT_INT4_CMP_ALL(T)                                                    \
  CUT_INT4_CMP(T, ==) CUT_INT4_CMP(T, !=) CUT_INT4_CMP(T, <)                   \
  CUT_INT4_CMP(T, <=) CUT_INT4_CMP(T, >) CUT_INT4_CMP(T, >=)
CUT_INT4_CMP_ALL(int4)
CUT_INT4_CMP_ALL(uint4)
#undef CUT_INT4_CMP_ALL
#undef CUT_INT4_CMP

__device__ __forceinline__ int4 operator-(int4 a) {
  return cut_mk_i4(-a.x, -a.y, -a.z, -a.w);
}

// Integer vector intrinsics (Int32 / UInt32 unary + binary paths).
__device__ __forceinline__ int cut_absi(int x) { return x < 0 ? -x : x; }
__device__ __forceinline__ int cut_signi(int x) { return (x > 0) - (x < 0); }
__device__ __forceinline__ int4 abs(int4 a) {
  return cut_mk_i4(cut_absi(a.x), cut_absi(a.y), cut_absi(a.z), cut_absi(a.w));
}
__device__ __forceinline__ int4 sign(int4 a) {
  return cut_mk_i4(cut_signi(a.x), cut_signi(a.y), cut_signi(a.z),
                   cut_signi(a.w));
}
#define CUT_IVEC_MINMAX(T, MK, name, cmp)                                      \
  __device__ __forceinline__ T name(T a, T b) {                                \
    return MK(a.x cmp b.x ? a.x : b.x, a.y cmp b.y ? a.y : b.y,                 \
             a.z cmp b.z ? a.z : b.z, a.w cmp b.w ? a.w : b.w);                \
  }
CUT_IVEC_MINMAX(int4, cut_mk_i4, min, <)
CUT_IVEC_MINMAX(int4, cut_mk_i4, max, >)
CUT_IVEC_MINMAX(uint4, cut_mk_u4, min, <)
CUT_IVEC_MINMAX(uint4, cut_mk_u4, max, >)
#undef CUT_IVEC_MINMAX
__device__ __forceinline__ int4 clamp(int4 a, int4 lo, int4 hi) {
  return min(max(a, lo), hi);
}
__device__ __forceinline__ uint4 clamp(uint4 a, uint4 lo, uint4 hi) {
  return min(max(a, lo), hi);
}

// ===========================================================================
// select() — HLSL ternary select(mask, a, b) returns a where mask!=0 else b.
// ===========================================================================
__device__ __forceinline__ float4 select(int4 c, float4 a, float4 b) {
  return cut_mk_f4(c.x ? a.x : b.x, c.y ? a.y : b.y, c.z ? a.z : b.z,
                   c.w ? a.w : b.w);
}
__device__ __forceinline__ int4 select(int4 c, int4 a, int4 b) {
  return cut_mk_i4(c.x ? a.x : b.x, c.y ? a.y : b.y, c.z ? a.z : b.z,
                   c.w ? a.w : b.w);
}
__device__ __forceinline__ uint4 select(int4 c, uint4 a, uint4 b) {
  return cut_mk_u4(c.x ? a.x : b.x, c.y ? a.y : b.y, c.z ? a.z : b.z,
                   c.w ? a.w : b.w);
}

// ===========================================================================
// half4 path: implemented by promoting to float4, computing, and demoting.
// Keeps half kernels compiling/correct without a full native-half intrinsic
// set; precision matches float compute rounded to half (acceptable for tests).
// ===========================================================================
__device__ __forceinline__ float4 cut_h2f(half4 a) {
  return cut_mk_f4(__half2float(a.x), __half2float(a.y), __half2float(a.z),
                   __half2float(a.w));
}
// HLSL "float4(half4-expr)" — lane-wise widening constructor.
__device__ __forceinline__ float4 cut_mk_f4(half4 v) { return cut_h2f(v); }
__device__ __forceinline__ half4 cut_f2h(float4 v) {
  half4 r;
  r.x = __float2half(v.x); r.y = __float2half(v.y);
  r.z = __float2half(v.z); r.w = __float2half(v.w);
  return r;
}

__device__ __forceinline__ half4 cut_cast_h4(float4 v) { return cut_f2h(v); }
__device__ __forceinline__ half4 cut_cast_h4(int4 v) {
  return cut_f2h(cut_cast_f4(v));
}
__device__ __forceinline__ float4 cut_cast_f4(half4 v) { return cut_h2f(v); }

#define CUT_H4_BINOP(op)                                                       \
  __device__ __forceinline__ half4 operator op(half4 a, half4 b) {             \
    return cut_f2h(cut_h2f(a) op cut_h2f(b));                                   \
  }                                                                            \
  __device__ __forceinline__ half4 operator op(half4 a, float b) {             \
    return cut_f2h(cut_h2f(a) op cut_cast_f4(b));                               \
  }                                                                            \
  __device__ __forceinline__ half4 operator op(float a, half4 b) {             \
    return cut_f2h(cut_cast_f4(a) op cut_h2f(b));                               \
  }                                                                            \
  __device__ __forceinline__ half4 operator op(half4 a, double b) {            \
    return cut_f2h(cut_h2f(a) op cut_cast_f4(b));                               \
  }                                                                            \
  __device__ __forceinline__ half4 operator op(double a, half4 b) {            \
    return cut_f2h(cut_cast_f4(a) op cut_h2f(b));                               \
  }
CUT_H4_BINOP(+)
CUT_H4_BINOP(-)
CUT_H4_BINOP(*)
CUT_H4_BINOP(/)
CUT_H4_BINOP(%)
#undef CUT_H4_BINOP

__device__ __forceinline__ half4 operator-(half4 a) {
  return cut_f2h(-cut_h2f(a));
}

#define CUT_H4_CMP(op)                                                         \
  __device__ __forceinline__ int4 operator op(half4 a, half4 b) {              \
    return cut_h2f(a) op cut_h2f(b);                                           \
  }                                                                            \
  __device__ __forceinline__ int4 operator op(half4 a, float b) {              \
    return cut_h2f(a) op cut_cast_f4(b);                                       \
  }
CUT_H4_CMP(==)
CUT_H4_CMP(!=)
CUT_H4_CMP(<)
CUT_H4_CMP(<=)
CUT_H4_CMP(>)
CUT_H4_CMP(>=)
#undef CUT_H4_CMP

// Unary intrinsics (delegate to the float4 overloads).
#define CUT_H4_UNARY(name)                                                     \
  __device__ __forceinline__ half4 name(half4 a) {                             \
    return cut_f2h(name(cut_h2f(a)));                                          \
  }
CUT_H4_UNARY(abs) CUT_H4_UNARY(sqrt) CUT_H4_UNARY(rsqrt) CUT_H4_UNARY(exp)
CUT_H4_UNARY(exp2) CUT_H4_UNARY(log) CUT_H4_UNARY(log2) CUT_H4_UNARY(sin)
CUT_H4_UNARY(cos) CUT_H4_UNARY(tan) CUT_H4_UNARY(asin) CUT_H4_UNARY(acos)
CUT_H4_UNARY(atan) CUT_H4_UNARY(sinh) CUT_H4_UNARY(cosh) CUT_H4_UNARY(tanh)
CUT_H4_UNARY(floor) CUT_H4_UNARY(ceil) CUT_H4_UNARY(trunc) CUT_H4_UNARY(sign)
CUT_H4_UNARY(frac) CUT_H4_UNARY(degrees) CUT_H4_UNARY(radians)
#undef CUT_H4_UNARY
__device__ __forceinline__ int4 isnan(half4 a) { return isnan(cut_h2f(a)); }
__device__ __forceinline__ int4 isinf(half4 a) { return isinf(cut_h2f(a)); }

// Binary / ternary intrinsics.
__device__ __forceinline__ half4 pow(half4 a, half4 b) {
  return cut_f2h(pow(cut_h2f(a), cut_h2f(b)));
}
__device__ __forceinline__ half4 atan2(half4 a, half4 b) {
  return cut_f2h(atan2(cut_h2f(a), cut_h2f(b)));
}
#define CUT_H4_MINMAX(name)                                                    \
  __device__ __forceinline__ half4 name(half4 a, half4 b) {                    \
    return cut_f2h(name(cut_h2f(a), cut_h2f(b)));                              \
  }                                                                            \
  __device__ __forceinline__ half4 name(half4 a, float b) {                    \
    return cut_f2h(name(cut_h2f(a), cut_cast_f4(b)));                          \
  }                                                                            \
  __device__ __forceinline__ half4 name(half4 a, double b) {                   \
    return cut_f2h(name(cut_h2f(a), cut_cast_f4(b)));                          \
  }
CUT_H4_MINMAX(min)
CUT_H4_MINMAX(max)
#undef CUT_H4_MINMAX
__device__ __forceinline__ half4 clamp(half4 a, half4 lo, half4 hi) {
  return cut_f2h(clamp(cut_h2f(a), cut_h2f(lo), cut_h2f(hi)));
}
__device__ __forceinline__ half4 clamp(half4 a, float lo, float hi) {
  return cut_f2h(clamp(cut_h2f(a), cut_cast_f4(lo), cut_cast_f4(hi)));
}
__device__ __forceinline__ half4 lerp(half4 a, half4 b, half4 t) {
  return cut_f2h(lerp(cut_h2f(a), cut_h2f(b), cut_h2f(t)));
}
__device__ __forceinline__ half4 select(int4 c, half4 a, half4 b) {
  return cut_f2h(select(c, cut_h2f(a), cut_h2f(b)));
}

// half bitwise via 16-bit reinterpret (asint16/asfloat16 in the HLSL bodies).
struct cut_s4 {
  unsigned short x, y, z, w;
};
__device__ __forceinline__ cut_s4 asint16(half4 a) {
  cut_s4 r;
  r.x = __half_as_ushort(a.x); r.y = __half_as_ushort(a.y);
  r.z = __half_as_ushort(a.z); r.w = __half_as_ushort(a.w);
  return r;
}
__device__ __forceinline__ half4 asfloat16(cut_s4 v) {
  half4 r;
  r.x = __ushort_as_half(v.x); r.y = __ushort_as_half(v.y);
  r.z = __ushort_as_half(v.z); r.w = __ushort_as_half(v.w);
  return r;
}
#define CUT_S4_BINOP(op)                                                       \
  __device__ __forceinline__ cut_s4 operator op(cut_s4 a, cut_s4 b) {          \
    cut_s4 r;                                                                  \
    r.x = a.x op b.x; r.y = a.y op b.y;                                        \
    r.z = a.z op b.z; r.w = a.w op b.w;                                        \
    return r;                                                                  \
  }
CUT_S4_BINOP(&)
CUT_S4_BINOP(|)
CUT_S4_BINOP(^)
#undef CUT_S4_BINOP
// 16-bit shifts mask the count to 4 bits (SPIR-V/HLSL half semantics).
#define CUT_S4_SHIFT(op)                                                       \
  __device__ __forceinline__ cut_s4 operator op(cut_s4 a, cut_s4 b) {          \
    cut_s4 r;                                                                  \
    r.x = a.x op(b.x & 0xF); r.y = a.y op(b.y & 0xF);                          \
    r.z = a.z op(b.z & 0xF); r.w = a.w op(b.w & 0xF);                          \
    return r;                                                                  \
  }
CUT_S4_SHIFT(<<)
CUT_S4_SHIFT(>>)
#undef CUT_S4_SHIFT
