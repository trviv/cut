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
// SSE Vector-Scalar Operations (128-bit, 4 floats at a time)
// =============================================================================
#if CUT_SIMD_SSE

inline void
sseAddScalar(const float *a, float scalar, float *out, size_t count) {
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_add_ps(va, vscalar);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] + scalar;
  }
}

inline void
sseSubScalar(const float *a, float scalar, float *out, size_t count) {
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_sub_ps(va, vscalar);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] - scalar;
  }
}

inline void
sseMulScalar(const float *a, float scalar, float *out, size_t count) {
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_mul_ps(va, vscalar);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] * scalar;
  }
}

inline void
sseDivScalar(const float *a, float scalar, float *out, size_t count) {
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_div_ps(va, vscalar);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] / scalar;
  }
}

inline void
sseMinScalar(const float *a, float scalar, float *out, size_t count) {
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_min_ps(va, vscalar);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < scalar ? a[i] : scalar;
  }
}

inline void
sseMaxScalar(const float *a, float scalar, float *out, size_t count) {
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_max_ps(va, vscalar);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > scalar ? a[i] : scalar;
  }
}

// SSE Vector-Scalar Comparison Operations
inline void
sseEqualScalar(const float *a, float scalar, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmpeq_ps(va, vscalar);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] == scalar ? 1.0f : 0.0f;
  }
}

inline void
sseNotEqualScalar(const float *a, float scalar, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmpneq_ps(va, vscalar);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] != scalar ? 1.0f : 0.0f;
  }
}

inline void
sseLessScalar(const float *a, float scalar, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmplt_ps(va, vscalar);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < scalar ? 1.0f : 0.0f;
  }
}

inline void
sseLessEqualScalar(const float *a, float scalar, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmple_ps(va, vscalar);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] <= scalar ? 1.0f : 0.0f;
  }
}

inline void
sseGreaterScalar(const float *a, float scalar, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmpgt_ps(va, vscalar);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > scalar ? 1.0f : 0.0f;
  }
}

inline void
sseGreaterEqualScalar(const float *a, float scalar, float *out, size_t count) {
  const __m128 one = _mm_set1_ps(1.0f);
  __m128 vscalar = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmpge_ps(va, vscalar);
    __m128 result = _mm_and_ps(cmp, one);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] >= scalar ? 1.0f : 0.0f;
  }
}

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

// =============================================================================
// AVX Vector-Scalar Operations (256-bit, 8 floats at a time)
// =============================================================================
#if CUT_SIMD_AVX

inline void
avxAddScalar(const float *a, float scalar, float *out, size_t count) {
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 result = _mm256_add_ps(va, vscalar256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_add_ps(va, vscalar128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] + scalar;
  }
}

inline void
avxSubScalar(const float *a, float scalar, float *out, size_t count) {
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 result = _mm256_sub_ps(va, vscalar256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_sub_ps(va, vscalar128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] - scalar;
  }
}

inline void
avxMulScalar(const float *a, float scalar, float *out, size_t count) {
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 result = _mm256_mul_ps(va, vscalar256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_mul_ps(va, vscalar128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] * scalar;
  }
}

inline void
avxDivScalar(const float *a, float scalar, float *out, size_t count) {
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 result = _mm256_div_ps(va, vscalar256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_div_ps(va, vscalar128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] / scalar;
  }
}

inline void
avxMinScalar(const float *a, float scalar, float *out, size_t count) {
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 result = _mm256_min_ps(va, vscalar256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_min_ps(va, vscalar128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < scalar ? a[i] : scalar;
  }
}

inline void
avxMaxScalar(const float *a, float scalar, float *out, size_t count) {
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 result = _mm256_max_ps(va, vscalar256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 result = _mm_max_ps(va, vscalar128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > scalar ? a[i] : scalar;
  }
}

// AVX Vector-Scalar Comparison Operations
inline void
avxEqualScalar(const float *a, float scalar, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 cmp = _mm256_cmp_ps(va, vscalar256, _CMP_EQ_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmpeq_ps(va, vscalar128);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] == scalar ? 1.0f : 0.0f;
  }
}

inline void
avxNotEqualScalar(const float *a, float scalar, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 cmp = _mm256_cmp_ps(va, vscalar256, _CMP_NEQ_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmpneq_ps(va, vscalar128);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] != scalar ? 1.0f : 0.0f;
  }
}

inline void
avxLessScalar(const float *a, float scalar, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 cmp = _mm256_cmp_ps(va, vscalar256, _CMP_LT_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmplt_ps(va, vscalar128);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < scalar ? 1.0f : 0.0f;
  }
}

inline void
avxLessEqualScalar(const float *a, float scalar, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 cmp = _mm256_cmp_ps(va, vscalar256, _CMP_LE_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmple_ps(va, vscalar128);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] <= scalar ? 1.0f : 0.0f;
  }
}

inline void
avxGreaterScalar(const float *a, float scalar, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 cmp = _mm256_cmp_ps(va, vscalar256, _CMP_GT_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmpgt_ps(va, vscalar128);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > scalar ? 1.0f : 0.0f;
  }
}

inline void
avxGreaterEqualScalar(const float *a, float scalar, float *out, size_t count) {
  const __m256 one256 = _mm256_set1_ps(1.0f);
  const __m128 one128 = _mm_set1_ps(1.0f);
  __m256 vscalar256 = _mm256_set1_ps(scalar);
  __m128 vscalar128 = _mm_set1_ps(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 cmp = _mm256_cmp_ps(va, vscalar256, _CMP_GE_OQ);
    __m256 result = _mm256_and_ps(cmp, one256);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 cmp = _mm_cmpge_ps(va, vscalar128);
    __m128 result = _mm_and_ps(cmp, one128);
    _mm_storeu_ps(out + i, result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] >= scalar ? 1.0f : 0.0f;
  }
}

#endif // CUT_SIMD_AVX

// =============================================================================
// SSE2 Integer Operations (128-bit, 4 int32s at a time)
// =============================================================================
#if CUT_SIMD_SSE2

// SSE2 Integer Binary Vec-Vec Operations
inline void
sseAddInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_add_epi32(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] + b[i];
  }
}

inline void
sseSubInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_sub_epi32(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] - b[i];
  }
}

#if CUT_SIMD_SSE4_1
inline void
sseMulInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_mullo_epi32(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] * b[i];
  }
}

inline void
sseMinInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_min_epi32(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < b[i] ? a[i] : b[i];
  }
}

inline void
sseMaxInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_max_epi32(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > b[i] ? a[i] : b[i];
  }
}
#endif // CUT_SIMD_SSE4_1

// SSE2 Integer Comparison Operations
inline void
sseEqualInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  const __m128i one = _mm_set1_epi32(1);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i cmp = _mm_cmpeq_epi32(va, vb);
    __m128i result = _mm_and_si128(cmp, one);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] == b[i] ? 1 : 0;
  }
}

inline void
sseGreaterInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  const __m128i one = _mm_set1_epi32(1);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i cmp = _mm_cmpgt_epi32(va, vb);
    __m128i result = _mm_and_si128(cmp, one);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > b[i] ? 1 : 0;
  }
}

inline void
sseLessInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  const __m128i one = _mm_set1_epi32(1);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i cmp = _mm_cmplt_epi32(va, vb);
    __m128i result = _mm_and_si128(cmp, one);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < b[i] ? 1 : 0;
  }
}

// SSE2 Integer Bitwise Operations
inline void sseBitwiseAndInt(const int32_t *a,
                             const int32_t *b,
                             int32_t *out,
                             size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_and_si128(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] & b[i];
  }
}

inline void sseBitwiseOrInt(const int32_t *a,
                            const int32_t *b,
                            int32_t *out,
                            size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_or_si128(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] | b[i];
  }
}

inline void sseBitwiseXorInt(const int32_t *a,
                             const int32_t *b,
                             int32_t *out,
                             size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_xor_si128(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] ^ b[i];
  }
}

inline void sseBitwiseNotInt(const int32_t *in, int32_t *out, size_t count) {
  const __m128i allOnes = _mm_set1_epi32(-1);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
    __m128i result = _mm_xor_si128(v, allOnes);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = ~in[i];
  }
}

// SSE2 Integer Vec-Scalar Operations
inline void
sseAddIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_add_epi32(va, vscalar);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] + scalar;
  }
}

inline void
sseSubIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_sub_epi32(va, vscalar);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] - scalar;
  }
}

#if CUT_SIMD_SSE4_1
inline void
sseMulIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_mullo_epi32(va, vscalar);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] * scalar;
  }
}

inline void
sseMinIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_min_epi32(va, vscalar);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < scalar ? a[i] : scalar;
  }
}

inline void
sseMaxIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_max_epi32(va, vscalar);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > scalar ? a[i] : scalar;
  }
}
#endif // CUT_SIMD_SSE4_1

inline void sseEqualIntScalar(const int32_t *a,
                              int32_t scalar,
                              int32_t *out,
                              size_t count) {
  const __m128i one = _mm_set1_epi32(1);
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i cmp = _mm_cmpeq_epi32(va, vscalar);
    __m128i result = _mm_and_si128(cmp, one);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] == scalar ? 1 : 0;
  }
}

inline void sseGreaterIntScalar(const int32_t *a,
                                int32_t scalar,
                                int32_t *out,
                                size_t count) {
  const __m128i one = _mm_set1_epi32(1);
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i cmp = _mm_cmpgt_epi32(va, vscalar);
    __m128i result = _mm_and_si128(cmp, one);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > scalar ? 1 : 0;
  }
}

inline void
sseLessIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  const __m128i one = _mm_set1_epi32(1);
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i cmp = _mm_cmplt_epi32(va, vscalar);
    __m128i result = _mm_and_si128(cmp, one);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < scalar ? 1 : 0;
  }
}

inline void sseBitwiseAndIntScalar(const int32_t *a,
                                   int32_t scalar,
                                   int32_t *out,
                                   size_t count) {
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_and_si128(va, vscalar);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] & scalar;
  }
}

inline void sseBitwiseOrIntScalar(const int32_t *a,
                                  int32_t scalar,
                                  int32_t *out,
                                  size_t count) {
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_or_si128(va, vscalar);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] | scalar;
  }
}

inline void sseBitwiseXorIntScalar(const int32_t *a,
                                   int32_t scalar,
                                   int32_t *out,
                                   size_t count) {
  __m128i vscalar = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_xor_si128(va, vscalar);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] ^ scalar;
  }
}

// SSE2 Integer Unary Operations
inline void sseNegInt(const int32_t *in, int32_t *out, size_t count) {
  const __m128i zero = _mm_setzero_si128();
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
    __m128i result = _mm_sub_epi32(zero, v);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = -in[i];
  }
}

#if CUT_SIMD_SSSE3
inline void sseAbsInt(const int32_t *in, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
    __m128i result = _mm_abs_epi32(v);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = in[i] < 0 ? -in[i] : in[i];
  }
}
#endif // CUT_SIMD_SSSE3

// SSE4.1 required for mullo_epi32
#if CUT_SIMD_SSE4_1
inline void sseSquareInt(const int32_t *in, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
    __m128i result = _mm_mullo_epi32(v, v);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = in[i] * in[i];
  }
}
#endif // CUT_SIMD_SSE4_1

#endif // CUT_SIMD_SSE2

// =============================================================================
// AVX2 Integer Operations (256-bit, 8 int32s at a time)
// =============================================================================
#if CUT_SIMD_AVX2

// AVX2 Integer Binary Vec-Vec Operations
inline void
avxAddInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i result = _mm256_add_epi32(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_add_epi32(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] + b[i];
  }
}

inline void
avxSubInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i result = _mm256_sub_epi32(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_sub_epi32(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] - b[i];
  }
}

inline void
avxMulInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i result = _mm256_mullo_epi32(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_mullo_epi32(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] * b[i];
  }
}

inline void
avxMinInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i result = _mm256_min_epi32(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_min_epi32(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < b[i] ? a[i] : b[i];
  }
}

inline void
avxMaxInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i result = _mm256_max_epi32(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_max_epi32(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > b[i] ? a[i] : b[i];
  }
}

// AVX2 Integer Comparison Operations
inline void
avxEqualInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  const __m256i one256 = _mm256_set1_epi32(1);
  const __m128i one128 = _mm_set1_epi32(1);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i cmp = _mm256_cmpeq_epi32(va, vb);
    __m256i result = _mm256_and_si256(cmp, one256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i cmp = _mm_cmpeq_epi32(va, vb);
    __m128i result = _mm_and_si128(cmp, one128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] == b[i] ? 1 : 0;
  }
}

inline void
avxGreaterInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  const __m256i one256 = _mm256_set1_epi32(1);
  const __m128i one128 = _mm_set1_epi32(1);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i cmp = _mm256_cmpgt_epi32(va, vb);
    __m256i result = _mm256_and_si256(cmp, one256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i cmp = _mm_cmpgt_epi32(va, vb);
    __m128i result = _mm_and_si128(cmp, one128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > b[i] ? 1 : 0;
  }
}

inline void
avxLessInt(const int32_t *a, const int32_t *b, int32_t *out, size_t count) {
  // a < b is equivalent to b > a
  avxGreaterInt(b, a, out, count);
}

// AVX2 Integer Bitwise Operations
inline void avxBitwiseAndInt(const int32_t *a,
                             const int32_t *b,
                             int32_t *out,
                             size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i result = _mm256_and_si256(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_and_si128(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] & b[i];
  }
}

inline void avxBitwiseOrInt(const int32_t *a,
                            const int32_t *b,
                            int32_t *out,
                            size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i result = _mm256_or_si256(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_or_si128(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] | b[i];
  }
}

inline void avxBitwiseXorInt(const int32_t *a,
                             const int32_t *b,
                             int32_t *out,
                             size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i result = _mm256_xor_si256(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i result = _mm_xor_si128(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] ^ b[i];
  }
}

inline void avxBitwiseNotInt(const int32_t *in, int32_t *out, size_t count) {
  const __m256i allOnes256 = _mm256_set1_epi32(-1);
  const __m128i allOnes128 = _mm_set1_epi32(-1);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));
    __m256i result = _mm256_xor_si256(v, allOnes256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
    __m128i result = _mm_xor_si128(v, allOnes128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = ~in[i];
  }
}

// AVX2 Integer Vec-Scalar Operations
inline void
avxAddIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i result = _mm256_add_epi32(va, vscalar256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_add_epi32(va, vscalar128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] + scalar;
  }
}

inline void
avxSubIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i result = _mm256_sub_epi32(va, vscalar256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_sub_epi32(va, vscalar128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] - scalar;
  }
}

inline void
avxMulIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i result = _mm256_mullo_epi32(va, vscalar256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_mullo_epi32(va, vscalar128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] * scalar;
  }
}

inline void
avxMinIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i result = _mm256_min_epi32(va, vscalar256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_min_epi32(va, vscalar128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < scalar ? a[i] : scalar;
  }
}

inline void
avxMaxIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i result = _mm256_max_epi32(va, vscalar256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_max_epi32(va, vscalar128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > scalar ? a[i] : scalar;
  }
}

inline void avxEqualIntScalar(const int32_t *a,
                              int32_t scalar,
                              int32_t *out,
                              size_t count) {
  const __m256i one256 = _mm256_set1_epi32(1);
  const __m128i one128 = _mm_set1_epi32(1);
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i cmp = _mm256_cmpeq_epi32(va, vscalar256);
    __m256i result = _mm256_and_si256(cmp, one256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i cmp = _mm_cmpeq_epi32(va, vscalar128);
    __m128i result = _mm_and_si128(cmp, one128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] == scalar ? 1 : 0;
  }
}

inline void avxGreaterIntScalar(const int32_t *a,
                                int32_t scalar,
                                int32_t *out,
                                size_t count) {
  const __m256i one256 = _mm256_set1_epi32(1);
  const __m128i one128 = _mm_set1_epi32(1);
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i cmp = _mm256_cmpgt_epi32(va, vscalar256);
    __m256i result = _mm256_and_si256(cmp, one256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i cmp = _mm_cmpgt_epi32(va, vscalar128);
    __m128i result = _mm_and_si128(cmp, one128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] > scalar ? 1 : 0;
  }
}

inline void
avxLessIntScalar(const int32_t *a, int32_t scalar, int32_t *out, size_t count) {
  // a < scalar is equivalent to NOT(a >= scalar) = NOT(a > scalar - 1) for
  // integers But simpler: just swap operands for greater than
  const __m256i one256 = _mm256_set1_epi32(1);
  const __m128i one128 = _mm_set1_epi32(1);
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    // a < scalar  =>  scalar > a
    __m256i cmp = _mm256_cmpgt_epi32(vscalar256, va);
    __m256i result = _mm256_and_si256(cmp, one256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i cmp = _mm_cmpgt_epi32(vscalar128, va);
    __m128i result = _mm_and_si128(cmp, one128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] < scalar ? 1 : 0;
  }
}

inline void avxBitwiseAndIntScalar(const int32_t *a,
                                   int32_t scalar,
                                   int32_t *out,
                                   size_t count) {
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i result = _mm256_and_si256(va, vscalar256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_and_si128(va, vscalar128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] & scalar;
  }
}

inline void avxBitwiseOrIntScalar(const int32_t *a,
                                  int32_t scalar,
                                  int32_t *out,
                                  size_t count) {
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i result = _mm256_or_si256(va, vscalar256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_or_si128(va, vscalar128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] | scalar;
  }
}

inline void avxBitwiseXorIntScalar(const int32_t *a,
                                   int32_t scalar,
                                   int32_t *out,
                                   size_t count) {
  __m256i vscalar256 = _mm256_set1_epi32(scalar);
  __m128i vscalar128 = _mm_set1_epi32(scalar);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i result = _mm256_xor_si256(va, vscalar256);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i result = _mm_xor_si128(va, vscalar128);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = a[i] ^ scalar;
  }
}

// AVX2 Integer Unary Operations
inline void avxNegInt(const int32_t *in, int32_t *out, size_t count) {
  const __m256i zero256 = _mm256_setzero_si256();
  const __m128i zero128 = _mm_setzero_si128();
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));
    __m256i result = _mm256_sub_epi32(zero256, v);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
    __m128i result = _mm_sub_epi32(zero128, v);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = -in[i];
  }
}

inline void avxAbsInt(const int32_t *in, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));
    __m256i result = _mm256_abs_epi32(v);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
    __m128i result = _mm_abs_epi32(v);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = in[i] < 0 ? -in[i] : in[i];
  }
}

inline void avxSquareInt(const int32_t *in, int32_t *out, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));
    __m256i result = _mm256_mullo_epi32(v, v);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i + 4 <= count; i += 4) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
    __m128i result = _mm_mullo_epi32(v, v);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
  }
  for (; i < count; ++i) {
    out[i] = in[i] * in[i];
  }
}

#endif // CUT_SIMD_AVX2

} // namespace simd
} // namespace cut
