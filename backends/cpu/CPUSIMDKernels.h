#pragma once

#include "CPUSIMDConfig.h"

#include <cmath>
#include <cstddef>

namespace cut {
namespace simd {

// =============================================================================
// SSE Implementation (128-bit, 4 floats at a time)
// =============================================================================
#if CUT_SIMD_SSE

// SSE Binary Operations
inline void sseAdd(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_add_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  // Handle remaining elements
  for (; i < count; ++i) {
    out[i] = a[i] + b[i];
  }
}

inline void sseSub(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_sub_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] - b[i];
  }
}

inline void sseMul(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_mul_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] * b[i];
  }
}

inline void sseDiv(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_div_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] / b[i];
  }
}

inline void sseMin(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_min_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < b[i] ? a[i] : b[i];
  }
}

inline void sseMax(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_max_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > b[i] ? a[i] : b[i];
  }
}

// SSE Comparison Operations (return 1.0f for true, 0.0f for false)
inline void sseEqual(const float *a, const float *b, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmpeq_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] == b[i] ? 1.0f : 0.0f;
  }
}

inline void
sseNotEqual(const float *a, const float *b, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmpneq_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] != b[i] ? 1.0f : 0.0f;
  }
}

inline void sseLess(const float *a, const float *b, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmplt_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < b[i] ? 1.0f : 0.0f;
  }
}

inline void
sseLessEqual(const float *a, const float *b, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmple_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] <= b[i] ? 1.0f : 0.0f;
  }
}

inline void
sseGreater(const float *a, const float *b, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmpgt_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > b[i] ? 1.0f : 0.0f;
  }
}

inline void
sseGreaterEqual(const float *a, const float *b, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmpge_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] >= b[i] ? 1.0f : 0.0f;
  }
}

// SSE Unary Operations
inline void sseNeg(const float *in, float *out, size_t count) {
  const __m128 signMask = _mm_set1_ps(-0.0f);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_xor_ps(v, signMask);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = -in[i];
  }
}

inline void sseAbs(const float *in, float *out, size_t count) {
  const __m128 absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_and_ps(v, absMask);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = std::abs(in[i]);
  }
}

inline void sseSqrt(const float *in, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_sqrt_ps(v);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = std::sqrt(in[i]);
  }
}

inline void sseReciprocal(const float *in, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_div_ps(one, v);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = 1.0f / in[i];
  }
}

inline void sseSquare(const float *in, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_mul_ps(v, v);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = in[i] * in[i];
  }
}

#if CUT_SIMD_SSE4_1
inline void sseFloor(const float *in, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_floor_ps(v);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = std::floor(in[i]);
  }
}

inline void sseCeil(const float *in, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_ceil_ps(v);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = std::ceil(in[i]);
  }
}

inline void sseRound(const float *in, float *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result =
        _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = std::round(in[i]);
  }
}
#endif // CUT_SIMD_SSE4_1

#endif // CUT_SIMD_SSE

// =============================================================================
// AVX Implementation (256-bit, 8 floats at a time)
// =============================================================================
#if CUT_SIMD_AVX

// AVX Binary Operations
inline void avxAdd(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 result = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(out + i, result);
  }
  // Handle remaining with SSE
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_add_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] + b[i];
  }
}

inline void avxSub(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 result = _mm256_sub_ps(va, vb);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_sub_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] - b[i];
  }
}

inline void avxMul(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 result = _mm256_mul_ps(va, vb);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_mul_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] * b[i];
  }
}

inline void avxDiv(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 result = _mm256_div_ps(va, vb);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_div_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] / b[i];
  }
}

inline void avxMin(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 result = _mm256_min_ps(va, vb);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_min_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < b[i] ? a[i] : b[i];
  }
}

inline void avxMax(const float *a, const float *b, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 result = _mm256_max_ps(va, vb);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 result = _mm_max_ps(va, vb);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > b[i] ? a[i] : b[i];
  }
}

// AVX Comparison Operations
inline void avxEqual(const float *a, const float *b, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_EQ_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmpeq_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] == b[i] ? 1.0f : 0.0f;
  }
}

inline void
avxNotEqual(const float *a, const float *b, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_NEQ_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmpneq_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] != b[i] ? 1.0f : 0.0f;
  }
}

inline void avxLess(const float *a, const float *b, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_LT_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmplt_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < b[i] ? 1.0f : 0.0f;
  }
}

inline void
avxLessEqual(const float *a, const float *b, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_LE_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmple_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] <= b[i] ? 1.0f : 0.0f;
  }
}

inline void
avxGreater(const float *a, const float *b, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_GT_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmpgt_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > b[i] ? 1.0f : 0.0f;
  }
}

inline void
avxGreaterEqual(const float *a, const float *b, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_GE_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 cmp = _mm_cmpge_ps(va, vb);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] >= b[i] ? 1.0f : 0.0f;
  }
}

// AVX Unary Operations
inline void avxNeg(const float *in, float *out, size_t count) {
  const __m256 signMask256 = _mm256_set1_ps(-0.0f);
  const __m128 signMask128 = _mm_set1_ps(-0.0f);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(in + i);
    __m256 result = _mm256_xor_ps(v, signMask256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_xor_ps(v, signMask128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = -in[i];
  }
}

inline void avxAbs(const float *in, float *out, size_t count) {
  const __m256 absMask256 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  const __m128 absMask128 = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(in + i);
    __m256 result = _mm256_and_ps(v, absMask256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_and_ps(v, absMask128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = std::abs(in[i]);
  }
}

inline void avxSqrt(const float *in, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(in + i);
    __m256 result = _mm256_sqrt_ps(v);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_sqrt_ps(v);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = std::sqrt(in[i]);
  }
}

inline void avxReciprocal(const float *in, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(in + i);
    __m256 result = _mm256_div_ps(one256, v);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_div_ps(one128, v);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = 1.0f / in[i];
  }
}

inline void avxSquare(const float *in, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(in + i);
    __m256 result = _mm256_mul_ps(v, v);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_mul_ps(v, v);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = in[i] * in[i];
  }
}

inline void avxFloor(const float *in, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(in + i);
    __m256 result = _mm256_floor_ps(v);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_floor_ps(v);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = std::floor(in[i]);
  }
}

inline void avxCeil(const float *in, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(in + i);
    __m256 result = _mm256_ceil_ps(v);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result = _mm_ceil_ps(v);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = std::ceil(in[i]);
  }
}

inline void avxRound(const float *in, float *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(in + i);
    __m256 result =
        _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(in + i);
    __m128 result =
        _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = std::round(in[i]);
  }
}

#endif // CUT_SIMD_AVX

} // namespace simd
} // namespace cut
