#include "CPUKernels.h"
#include "CPUSIMDConfig.h"
#include "CPUSIMDKernels.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace cut {

SIMDMode getEffectiveSIMDMode(SIMDMode requested) {
  if (requested == SIMDMode::Auto) {
#if CUT_SIMD_AVX
    return SIMDMode::AVX;
#elif CUT_SIMD_SSE
    return SIMDMode::SSE;
#else
    return SIMDMode::Scalar;
#endif
  }

  if (requested == SIMDMode::AVX) {
#if CUT_SIMD_AVX
    return SIMDMode::AVX;
#elif CUT_SIMD_SSE
    return SIMDMode::SSE;
#else
    return SIMDMode::Scalar;
#endif
  }

  if (requested == SIMDMode::SSE) {
#if CUT_SIMD_SSE
    return SIMDMode::SSE;
#else
    return SIMDMode::Scalar;
#endif
  }

  return SIMDMode::Scalar;
}

// =============================================================================
// Type-Generic Operation Helpers
// =============================================================================

namespace {

// Generic loop helpers
template <typename T, typename BinaryOp>
void binaryLoop(
    const T *a, const T *b, T *out, size_t start, size_t end, BinaryOp op) {
  for (size_t i = start; i < end; ++i) {
    out[i] = op(a[i], b[i]);
  }
}

template <typename T, typename UnaryOp>
void unaryLoop(const T *in, T *out, size_t start, size_t end, UnaryOp op) {
  for (size_t i = start; i < end; ++i) {
    out[i] = op(in[i]);
  }
}

template <typename T, typename BinaryOp>
void binaryVecScalarLoop(
    const T *a, T scalar, T *out, size_t start, size_t end, BinaryOp op) {
  for (size_t i = start; i < end; ++i) {
    out[i] = op(a[i], scalar);
  }
}

// =============================================================================
// Type-specific operation implementations using if constexpr
// =============================================================================

template <typename T>
inline T opAdd(T a, T b) {
  return a + b;
}

template <typename T>
inline T opSub(T a, T b) {
  return a - b;
}

template <typename T>
inline T opMul(T a, T b) {
  return a * b;
}

template <typename T>
inline T opDiv(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return b != 0 ? a / b : 0;
  } else {
    return a / b;
  }
}

template <typename T>
inline T opMod(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return b != 0 ? a % b : 0;
  } else {
    return a - b * std::floor(a / b);
  }
}

template <typename T>
inline T opFloorDiv(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    if (b == 0)
      return 0;
    T q = a / b;
    T r = a % b;
    if ((r != 0) && ((r < 0) != (b < 0))) {
      q -= 1;
    }
    return q;
  } else {
    return std::floor(a / b);
  }
}

template <typename T>
inline T opEqual(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a == b ? 1 : 0;
  } else {
    return a == b ? 1.0f : 0.0f;
  }
}

template <typename T>
inline T opNotEqual(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a != b ? 1 : 0;
  } else {
    return a != b ? 1.0f : 0.0f;
  }
}

template <typename T>
inline T opLess(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a < b ? 1 : 0;
  } else {
    return a < b ? 1.0f : 0.0f;
  }
}

template <typename T>
inline T opLessEqual(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a <= b ? 1 : 0;
  } else {
    return a <= b ? 1.0f : 0.0f;
  }
}

template <typename T>
inline T opGreater(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a > b ? 1 : 0;
  } else {
    return a > b ? 1.0f : 0.0f;
  }
}

template <typename T>
inline T opGreaterEqual(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a >= b ? 1 : 0;
  } else {
    return a >= b ? 1.0f : 0.0f;
  }
}

template <typename T>
inline T opMin(T a, T b) {
  return std::min(a, b);
}

template <typename T>
inline T opMax(T a, T b) {
  return std::max(a, b);
}

template <typename T>
inline T opBitwiseAnd(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a & b;
  } else {
    return static_cast<T>(static_cast<int32_t>(a) & static_cast<int32_t>(b));
  }
}

template <typename T>
inline T opBitwiseOr(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a | b;
  } else {
    return static_cast<T>(static_cast<int32_t>(a) | static_cast<int32_t>(b));
  }
}

template <typename T>
inline T opBitwiseXor(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a ^ b;
  } else {
    return static_cast<T>(static_cast<int32_t>(a) ^ static_cast<int32_t>(b));
  }
}

template <typename T>
inline T opLeftShift(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a << b;
  } else {
    return static_cast<T>(static_cast<int32_t>(a) << static_cast<int32_t>(b));
  }
}

template <typename T>
inline T opRightShift(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a >> b;
  } else {
    return static_cast<T>(static_cast<int32_t>(a) >> static_cast<int32_t>(b));
  }
}

template <typename T>
inline T opLogicalAnd(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return (a != 0 && b != 0) ? 1 : 0;
  } else {
    return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
  }
}

template <typename T>
inline T opLogicalOr(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return (a != 0 || b != 0) ? 1 : 0;
  } else {
    return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
  }
}

template <typename T>
inline T opLogicalXor(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return ((a != 0) != (b != 0)) ? 1 : 0;
  } else {
    return ((a != 0.0f) != (b != 0.0f)) ? 1.0f : 0.0f;
  }
}

// Float-only binary operations
inline float opPow(float a, float b) {
  return std::pow(a, b);
}
inline float opAtan2(float a, float b) {
  return std::atan2(a, b);
}
inline float opHypot(float a, float b) {
  return std::hypot(a, b);
}
inline float opCopysign(float a, float b) {
  return std::copysign(a, b);
}
inline float opFmod(float a, float b) {
  return std::fmod(a, b);
}
inline float opLeakyRelu(float x, float alpha) {
  return x > 0.0f ? x : alpha * x;
}

// Unary operations
template <typename T>
inline T opNeg(T x) {
  return -x;
}

template <typename T>
inline T opAbs(T x) {
  return std::abs(x);
}

template <typename T>
inline T opSign(T x) {
  if constexpr (std::is_integral_v<T>) {
    return x > 0 ? 1 : (x < 0 ? -1 : 0);
  } else {
    return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
  }
}

template <typename T>
inline T opSquare(T x) {
  return x * x;
}

template <typename T>
inline T opBitwiseNot(T x) {
  if constexpr (std::is_integral_v<T>) {
    return ~x;
  } else {
    return static_cast<T>(~static_cast<int32_t>(x));
  }
}

template <typename T>
inline T opLogicalNot(T x) {
  if constexpr (std::is_integral_v<T>) {
    return x == 0 ? 1 : 0;
  } else {
    return x == 0.0f ? 1.0f : 0.0f;
  }
}

// Float-only unary operations
inline float opSqrt(float x) {
  return std::sqrt(x);
}
inline float opExp(float x) {
  return std::exp(x);
}
inline float opExp2(float x) {
  return std::exp2(x);
}
inline float opExpm1(float x) {
  return std::expm1(x);
}
inline float opLog(float x) {
  return std::log(x);
}
inline float opLog2(float x) {
  return std::log2(x);
}
inline float opLog10(float x) {
  return std::log10(x);
}
inline float opLog1p(float x) {
  return std::log1p(x);
}
inline float opSin(float x) {
  return std::sin(x);
}
inline float opCos(float x) {
  return std::cos(x);
}
inline float opTan(float x) {
  return std::tan(x);
}
inline float opAsin(float x) {
  return std::asin(x);
}
inline float opAcos(float x) {
  return std::acos(x);
}
inline float opAtan(float x) {
  return std::atan(x);
}
inline float opSinh(float x) {
  return std::sinh(x);
}
inline float opCosh(float x) {
  return std::cosh(x);
}
inline float opTanh(float x) {
  return std::tanh(x);
}
inline float opFloor(float x) {
  return std::floor(x);
}
inline float opCeil(float x) {
  return std::ceil(x);
}
inline float opRound(float x) {
  return std::round(x);
}
inline float opCbrt(float x) {
  return std::cbrt(x);
}
inline float opReciprocal(float x) {
  return 1.0f / x;
}
inline float opDegrees(float x) {
  return x * 180.0f / 3.14159265358979323846f;
}
inline float opRadians(float x) {
  return x * 3.14159265358979323846f / 180.0f;
}
inline float opRelu(float x) {
  return x > 0.0f ? x : 0.0f;
}
inline float opSigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}
inline float opGelu(float x) {
  const float sqrt2pi = 0.7978845608028654f;
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + std::tanh(sqrt2pi * (x + 0.044715f * x3)));
}
inline float opSilu(float x) {
  return x / (1.0f + std::exp(-x));
}
inline float opSoftplus(float x) {
  return std::log(1.0f + std::exp(x));
}
inline float opIsNan(float x) {
  return std::isnan(x) ? 1.0f : 0.0f;
}
inline float opIsInf(float x) {
  return std::isinf(x) ? 1.0f : 0.0f;
}

} // namespace

// =============================================================================
// Template Implementation: executeBinaryKernel
// =============================================================================

template <typename T>
void executeBinaryKernel(OperatorEnum op,
                         const T *a,
                         const T *b,
                         T *out,
                         size_t start,
                         size_t end,
                         SIMDMode simdMode) {
  // SIMD path only for float
  if constexpr (std::is_same_v<T, float>) {
    const float *aStart = a + start;
    const float *bStart = b + start;
    float *outStart = out + start;
    size_t count = end - start;
    SIMDMode effectiveMode = getEffectiveSIMDMode(simdMode);

    switch (op) {
    case BinaryVecVecAdd:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxAdd(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseAdd(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecSub:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxSub(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseSub(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecMul:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMul(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMul(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecDiv:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxDiv(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseDiv(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecEqual:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxEqual(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseEqual(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecNotEqual:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxNotEqual(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseNotEqual(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecLess:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxLess(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseLess(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecLessEqual:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxLessEqual(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseLessEqual(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecGreater:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxGreater(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseGreater(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecGreaterEqual:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxGreaterEqual(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseGreaterEqual(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecMin:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMin(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMin(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecMax:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMax(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMax(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    default:
      break;
    }
  }

  // SIMD path for int32_t
  if constexpr (std::is_same_v<T, int32_t>) {
    const int32_t *aStart = a + start;
    const int32_t *bStart = b + start;
    int32_t *outStart = out + start;
    size_t count = end - start;
    SIMDMode effectiveMode = getEffectiveSIMDMode(simdMode);

    switch (op) {
    case BinaryVecVecAdd:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxAddInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseAddInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecSub:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxSubInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseSubInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecMul:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMulInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE4_1
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMulInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecMin:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMinInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE4_1
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMinInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecMax:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMaxInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE4_1
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMaxInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecEqual:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxEqualInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseEqualInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecLess:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxLessInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseLessInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecGreater:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxGreaterInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseGreaterInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecBitwiseAnd:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxBitwiseAndInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseBitwiseAndInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecBitwiseOr:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxBitwiseOrInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseBitwiseOrInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecVecBitwiseXor:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxBitwiseXorInt(aStart, bStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseBitwiseXorInt(aStart, bStart, outStart, count);
        return;
      }
#endif
      break;
    default:
      break;
    }
  }

  // Scalar fallback for all types
  switch (op) {
  // Arithmetic (all types)
  case BinaryVecVecAdd:
    binaryLoop(a, b, out, start, end, opAdd<T>);
    break;
  case BinaryVecVecSub:
    binaryLoop(a, b, out, start, end, opSub<T>);
    break;
  case BinaryVecVecMul:
    binaryLoop(a, b, out, start, end, opMul<T>);
    break;
  case BinaryVecVecDiv:
    binaryLoop(a, b, out, start, end, opDiv<T>);
    break;
  case BinaryVecVecMod:
    binaryLoop(a, b, out, start, end, opMod<T>);
    break;
  case BinaryVecVecFloorDiv:
    binaryLoop(a, b, out, start, end, opFloorDiv<T>);
    break;
  // Comparison (all types)
  case BinaryVecVecEqual:
    binaryLoop(a, b, out, start, end, opEqual<T>);
    break;
  case BinaryVecVecNotEqual:
    binaryLoop(a, b, out, start, end, opNotEqual<T>);
    break;
  case BinaryVecVecLess:
    binaryLoop(a, b, out, start, end, opLess<T>);
    break;
  case BinaryVecVecLessEqual:
    binaryLoop(a, b, out, start, end, opLessEqual<T>);
    break;
  case BinaryVecVecGreater:
    binaryLoop(a, b, out, start, end, opGreater<T>);
    break;
  case BinaryVecVecGreaterEqual:
    binaryLoop(a, b, out, start, end, opGreaterEqual<T>);
    break;
  case BinaryVecVecMin:
    binaryLoop(a, b, out, start, end, opMin<T>);
    break;
  case BinaryVecVecMax:
    binaryLoop(a, b, out, start, end, opMax<T>);
    break;
  // Bitwise (all types)
  case BinaryVecVecBitwiseAnd:
    binaryLoop(a, b, out, start, end, opBitwiseAnd<T>);
    break;
  case BinaryVecVecBitwiseOr:
    binaryLoop(a, b, out, start, end, opBitwiseOr<T>);
    break;
  case BinaryVecVecBitwiseXor:
    binaryLoop(a, b, out, start, end, opBitwiseXor<T>);
    break;
  case BinaryVecVecLeftShift:
    binaryLoop(a, b, out, start, end, opLeftShift<T>);
    break;
  case BinaryVecVecRightShift:
    binaryLoop(a, b, out, start, end, opRightShift<T>);
    break;
  // Logical (all types)
  case BinaryVecVecLogicalAnd:
    binaryLoop(a, b, out, start, end, opLogicalAnd<T>);
    break;
  case BinaryVecVecLogicalOr:
    binaryLoop(a, b, out, start, end, opLogicalOr<T>);
    break;
  case BinaryVecVecLogicalXor:
    binaryLoop(a, b, out, start, end, opLogicalXor<T>);
    break;
  // Float-only operations
  case BinaryVecVecPow:
  case BinaryVecVecAtan2:
  case BinaryVecVecHypot:
  case BinaryVecVecCopysign:
  case BinaryVecVecFmod:
    if constexpr (std::is_same_v<T, float>) {
      switch (op) {
      case BinaryVecVecPow:
        binaryLoop(a, b, out, start, end, opPow);
        break;
      case BinaryVecVecAtan2:
        binaryLoop(a, b, out, start, end, opAtan2);
        break;
      case BinaryVecVecHypot:
        binaryLoop(a, b, out, start, end, opHypot);
        break;
      case BinaryVecVecCopysign:
        binaryLoop(a, b, out, start, end, opCopysign);
        break;
      case BinaryVecVecFmod:
        binaryLoop(a, b, out, start, end, opFmod);
        break;
      default:
        break;
      }
    } else {
      throw std::runtime_error("Operator not supported for Int32 type");
    }
    break;
  default:
    throw std::runtime_error("Unknown binary operator");
  }
}

// =============================================================================
// Template Implementation: executeBinaryVecScalarKernel
// =============================================================================

template <typename T>
void executeBinaryVecScalarKernel(OperatorEnum op,
                                  const T *a,
                                  T scalar,
                                  T *out,
                                  size_t start,
                                  size_t end,
                                  SIMDMode simdMode) {
  // SIMD path only for float
  if constexpr (std::is_same_v<T, float>) {
    const float *aStart = a + start;
    float *outStart = out + start;
    size_t count = end - start;
    SIMDMode effectiveMode = getEffectiveSIMDMode(simdMode);

    switch (op) {
    case BinaryVecScalarAdd:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxAddScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseAddScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarSub:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxSubScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseSubScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarMul:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMulScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMulScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarDiv:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxDivScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseDivScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarMin:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMinScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMinScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarMax:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMaxScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMaxScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarEqual:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxEqualScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseEqualScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarNotEqual:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxNotEqualScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseNotEqualScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarLess:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxLessScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseLessScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarLessEqual:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxLessEqualScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseLessEqualScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarGreater:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxGreaterScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseGreaterScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarGreaterEqual:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxGreaterEqualScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseGreaterEqualScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    default:
      break;
    }
  }

  // SIMD path for int32_t
  if constexpr (std::is_same_v<T, int32_t>) {
    const int32_t *aStart = a + start;
    int32_t *outStart = out + start;
    size_t count = end - start;
    SIMDMode effectiveMode = getEffectiveSIMDMode(simdMode);

    switch (op) {
    case BinaryVecScalarAdd:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxAddIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseAddIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarSub:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxSubIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseSubIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarMul:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMulIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE4_1
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMulIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarMin:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMinIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE4_1
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMinIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarMax:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxMaxIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE4_1
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseMaxIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarEqual:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxEqualIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseEqualIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarLess:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxLessIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseLessIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarGreater:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxGreaterIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseGreaterIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarBitwiseAnd:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxBitwiseAndIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseBitwiseAndIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarBitwiseOr:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxBitwiseOrIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseBitwiseOrIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    case BinaryVecScalarBitwiseXor:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxBitwiseXorIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseBitwiseXorIntScalar(aStart, scalar, outStart, count);
        return;
      }
#endif
      break;
    default:
      break;
    }
  }

  // Scalar fallback for all types
  switch (op) {
  // Arithmetic (all types)
  case BinaryVecScalarAdd:
    binaryVecScalarLoop(a, scalar, out, start, end, opAdd<T>);
    break;
  case BinaryVecScalarSub:
    binaryVecScalarLoop(a, scalar, out, start, end, opSub<T>);
    break;
  case BinaryVecScalarMul:
    binaryVecScalarLoop(a, scalar, out, start, end, opMul<T>);
    break;
  case BinaryVecScalarDiv:
    binaryVecScalarLoop(a, scalar, out, start, end, opDiv<T>);
    break;
  case BinaryVecScalarMod:
    binaryVecScalarLoop(a, scalar, out, start, end, opMod<T>);
    break;
  case BinaryVecScalarFloorDiv:
    binaryVecScalarLoop(a, scalar, out, start, end, opFloorDiv<T>);
    break;
  // Comparison (all types)
  case BinaryVecScalarEqual:
    binaryVecScalarLoop(a, scalar, out, start, end, opEqual<T>);
    break;
  case BinaryVecScalarNotEqual:
    binaryVecScalarLoop(a, scalar, out, start, end, opNotEqual<T>);
    break;
  case BinaryVecScalarLess:
    binaryVecScalarLoop(a, scalar, out, start, end, opLess<T>);
    break;
  case BinaryVecScalarLessEqual:
    binaryVecScalarLoop(a, scalar, out, start, end, opLessEqual<T>);
    break;
  case BinaryVecScalarGreater:
    binaryVecScalarLoop(a, scalar, out, start, end, opGreater<T>);
    break;
  case BinaryVecScalarGreaterEqual:
    binaryVecScalarLoop(a, scalar, out, start, end, opGreaterEqual<T>);
    break;
  case BinaryVecScalarMin:
    binaryVecScalarLoop(a, scalar, out, start, end, opMin<T>);
    break;
  case BinaryVecScalarMax:
    binaryVecScalarLoop(a, scalar, out, start, end, opMax<T>);
    break;
  // Bitwise (all types)
  case BinaryVecScalarBitwiseAnd:
    binaryVecScalarLoop(a, scalar, out, start, end, opBitwiseAnd<T>);
    break;
  case BinaryVecScalarBitwiseOr:
    binaryVecScalarLoop(a, scalar, out, start, end, opBitwiseOr<T>);
    break;
  case BinaryVecScalarBitwiseXor:
    binaryVecScalarLoop(a, scalar, out, start, end, opBitwiseXor<T>);
    break;
  case BinaryVecScalarLeftShift:
    binaryVecScalarLoop(a, scalar, out, start, end, opLeftShift<T>);
    break;
  case BinaryVecScalarRightShift:
    binaryVecScalarLoop(a, scalar, out, start, end, opRightShift<T>);
    break;
  // Logical (all types)
  case BinaryVecScalarLogicalAnd:
    binaryVecScalarLoop(a, scalar, out, start, end, opLogicalAnd<T>);
    break;
  case BinaryVecScalarLogicalOr:
    binaryVecScalarLoop(a, scalar, out, start, end, opLogicalOr<T>);
    break;
  case BinaryVecScalarLogicalXor:
    binaryVecScalarLoop(a, scalar, out, start, end, opLogicalXor<T>);
    break;
  // Float-only operations
  case BinaryVecScalarPow:
  case BinaryVecScalarAtan2:
  case BinaryVecScalarHypot:
  case BinaryVecScalarCopysign:
  case BinaryVecScalarFmod:
  case BinaryVecScalarLeakyRelu:
    if constexpr (std::is_same_v<T, float>) {
      switch (op) {
      case BinaryVecScalarPow:
        binaryVecScalarLoop(a, scalar, out, start, end, opPow);
        break;
      case BinaryVecScalarAtan2:
        binaryVecScalarLoop(a, scalar, out, start, end, opAtan2);
        break;
      case BinaryVecScalarHypot:
        binaryVecScalarLoop(a, scalar, out, start, end, opHypot);
        break;
      case BinaryVecScalarCopysign:
        binaryVecScalarLoop(a, scalar, out, start, end, opCopysign);
        break;
      case BinaryVecScalarFmod:
        binaryVecScalarLoop(a, scalar, out, start, end, opFmod);
        break;
      case BinaryVecScalarLeakyRelu:
        binaryVecScalarLoop(a, scalar, out, start, end, opLeakyRelu);
        break;
      default:
        break;
      }
    } else {
      throw std::runtime_error("Operator not supported for Int32 type");
    }
    break;
  default:
    throw std::runtime_error("Unknown binary vec-scalar operator");
  }
}

// =============================================================================
// Template Implementation: executeUnaryKernel
// =============================================================================

template <typename T>
void executeUnaryKernel(OperatorEnum op,
                        const T *in,
                        T *out,
                        size_t start,
                        size_t end,
                        SIMDMode simdMode) {
  // SIMD path only for float
  if constexpr (std::is_same_v<T, float>) {
    const float *inStart = in + start;
    float *outStart = out + start;
    size_t count = end - start;
    SIMDMode effectiveMode = getEffectiveSIMDMode(simdMode);

    switch (op) {
    case UnaryNeg:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxNeg(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseNeg(inStart, outStart, count);
        return;
      }
#endif
      break;
    case UnaryAbs:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxAbs(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseAbs(inStart, outStart, count);
        return;
      }
#endif
      break;
    case UnarySqrt:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxSqrt(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseSqrt(inStart, outStart, count);
        return;
      }
#endif
      break;
    case UnaryFloor:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxFloor(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE4_1
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseFloor(inStart, outStart, count);
        return;
      }
#endif
      break;
    case UnaryCeil:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxCeil(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE4_1
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseCeil(inStart, outStart, count);
        return;
      }
#endif
      break;
    case UnaryRound:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxRound(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE4_1
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseRound(inStart, outStart, count);
        return;
      }
#endif
      break;
    case UnaryReciprocal:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxReciprocal(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseReciprocal(inStart, outStart, count);
        return;
      }
#endif
      break;
    case UnarySquare:
#if CUT_SIMD_AVX
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxSquare(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseSquare(inStart, outStart, count);
        return;
      }
#endif
      break;
    default:
      break;
    }
  }
  // SIMD path for int32
  else if constexpr (std::is_same_v<T, int32_t>) {
    const int32_t *inStart = in + start;
    int32_t *outStart = out + start;
    size_t count = end - start;
    SIMDMode effectiveMode = getEffectiveSIMDMode(simdMode);

    switch (op) {
    case UnaryNeg:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxNegInt(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseNegInt(inStart, outStart, count);
        return;
      }
#endif
      break;
    case UnaryAbs:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxAbsInt(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSSE3
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseAbsInt(inStart, outStart, count);
        return;
      }
#endif
      break;
    case UnaryBitwiseNot:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxBitwiseNotInt(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE2
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseBitwiseNotInt(inStart, outStart, count);
        return;
      }
#endif
      break;
    case UnarySquare:
#if CUT_SIMD_AVX2
      if (effectiveMode == SIMDMode::AVX) {
        simd::avxSquareInt(inStart, outStart, count);
        return;
      }
#endif
#if CUT_SIMD_SSE4_1
      if (effectiveMode == SIMDMode::SSE) {
        simd::sseSquareInt(inStart, outStart, count);
        return;
      }
#endif
      break;
    default:
      break;
    }
  }

  // Scalar fallback for all types
  switch (op) {
  // Operations supported by all types
  case UnaryNeg:
    unaryLoop(in, out, start, end, opNeg<T>);
    break;
  case UnaryAbs:
    unaryLoop(in, out, start, end, opAbs<T>);
    break;
  case UnarySign:
    unaryLoop(in, out, start, end, opSign<T>);
    break;
  case UnarySquare:
    unaryLoop(in, out, start, end, opSquare<T>);
    break;
  case UnaryBitwiseNot:
    unaryLoop(in, out, start, end, opBitwiseNot<T>);
    break;
  case UnaryLogicalNot:
    unaryLoop(in, out, start, end, opLogicalNot<T>);
    break;
  // Float-only operations
  case UnarySqrt:
  case UnaryExp:
  case UnaryExp2:
  case UnaryExpm1:
  case UnaryLog:
  case UnaryLog2:
  case UnaryLog10:
  case UnaryLog1p:
  case UnarySin:
  case UnaryCos:
  case UnaryTan:
  case UnaryAsin:
  case UnaryAcos:
  case UnaryAtan:
  case UnarySinh:
  case UnaryCosh:
  case UnaryTanh:
  case UnaryFloor:
  case UnaryCeil:
  case UnaryRound:
  case UnaryCbrt:
  case UnaryReciprocal:
  case UnaryDegrees:
  case UnaryRadians:
  case UnaryRelu:
  case UnarySigmoid:
  case UnaryGelu:
  case UnarySilu:
  case UnarySoftplus:
  case UnaryIsNan:
  case UnaryIsInf:
    if constexpr (std::is_same_v<T, float>) {
      switch (op) {
      case UnarySqrt:
        unaryLoop(in, out, start, end, opSqrt);
        break;
      case UnaryExp:
        unaryLoop(in, out, start, end, opExp);
        break;
      case UnaryExp2:
        unaryLoop(in, out, start, end, opExp2);
        break;
      case UnaryExpm1:
        unaryLoop(in, out, start, end, opExpm1);
        break;
      case UnaryLog:
        unaryLoop(in, out, start, end, opLog);
        break;
      case UnaryLog2:
        unaryLoop(in, out, start, end, opLog2);
        break;
      case UnaryLog10:
        unaryLoop(in, out, start, end, opLog10);
        break;
      case UnaryLog1p:
        unaryLoop(in, out, start, end, opLog1p);
        break;
      case UnarySin:
        unaryLoop(in, out, start, end, opSin);
        break;
      case UnaryCos:
        unaryLoop(in, out, start, end, opCos);
        break;
      case UnaryTan:
        unaryLoop(in, out, start, end, opTan);
        break;
      case UnaryAsin:
        unaryLoop(in, out, start, end, opAsin);
        break;
      case UnaryAcos:
        unaryLoop(in, out, start, end, opAcos);
        break;
      case UnaryAtan:
        unaryLoop(in, out, start, end, opAtan);
        break;
      case UnarySinh:
        unaryLoop(in, out, start, end, opSinh);
        break;
      case UnaryCosh:
        unaryLoop(in, out, start, end, opCosh);
        break;
      case UnaryTanh:
        unaryLoop(in, out, start, end, opTanh);
        break;
      case UnaryFloor:
        unaryLoop(in, out, start, end, opFloor);
        break;
      case UnaryCeil:
        unaryLoop(in, out, start, end, opCeil);
        break;
      case UnaryRound:
        unaryLoop(in, out, start, end, opRound);
        break;
      case UnaryCbrt:
        unaryLoop(in, out, start, end, opCbrt);
        break;
      case UnaryReciprocal:
        unaryLoop(in, out, start, end, opReciprocal);
        break;
      case UnaryDegrees:
        unaryLoop(in, out, start, end, opDegrees);
        break;
      case UnaryRadians:
        unaryLoop(in, out, start, end, opRadians);
        break;
      case UnaryRelu:
        unaryLoop(in, out, start, end, opRelu);
        break;
      case UnarySigmoid:
        unaryLoop(in, out, start, end, opSigmoid);
        break;
      case UnaryGelu:
        unaryLoop(in, out, start, end, opGelu);
        break;
      case UnarySilu:
        unaryLoop(in, out, start, end, opSilu);
        break;
      case UnarySoftplus:
        unaryLoop(in, out, start, end, opSoftplus);
        break;
      case UnaryIsNan:
        unaryLoop(in, out, start, end, opIsNan);
        break;
      case UnaryIsInf:
        unaryLoop(in, out, start, end, opIsInf);
        break;
      default:
        break;
      }
    } else {
      throw std::runtime_error("Operator not supported for Int32 type");
    }
    break;
  default:
    throw std::runtime_error("Unknown unary operator");
  }
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template void executeBinaryKernel<float>(OperatorEnum op,
                                         const float *a,
                                         const float *b,
                                         float *out,
                                         size_t start,
                                         size_t end,
                                         SIMDMode simdMode);

template void executeBinaryKernel<int32_t>(OperatorEnum op,
                                           const int32_t *a,
                                           const int32_t *b,
                                           int32_t *out,
                                           size_t start,
                                           size_t end,
                                           SIMDMode simdMode);

template void executeBinaryVecScalarKernel<float>(OperatorEnum op,
                                                  const float *a,
                                                  float scalar,
                                                  float *out,
                                                  size_t start,
                                                  size_t end,
                                                  SIMDMode simdMode);

template void executeBinaryVecScalarKernel<int32_t>(OperatorEnum op,
                                                    const int32_t *a,
                                                    int32_t scalar,
                                                    int32_t *out,
                                                    size_t start,
                                                    size_t end,
                                                    SIMDMode simdMode);

template void executeUnaryKernel<float>(OperatorEnum op,
                                        const float *in,
                                        float *out,
                                        size_t start,
                                        size_t end,
                                        SIMDMode simdMode);

template void executeUnaryKernel<int32_t>(OperatorEnum op,
                                          const int32_t *in,
                                          int32_t *out,
                                          size_t start,
                                          size_t end,
                                          SIMDMode simdMode);

// =============================================================================
// Non-templated functions (ternary, reduction, matrix ops)
// =============================================================================

void executeTernaryClampKernel(const float *in,
                               float minVal,
                               float maxVal,
                               float *out,
                               size_t start,
                               size_t end,
                               SIMDMode simdMode) {
  (void)simdMode;
  for (size_t i = start; i < end; ++i) {
    float val = in[i];
    if (val < minVal) {
      out[i] = minVal;
    } else if (val > maxVal) {
      out[i] = maxVal;
    } else {
      out[i] = val;
    }
  }
}

void executeReductionKernel(OperatorEnum op,
                            const float *in,
                            float *out,
                            size_t count,
                            SIMDMode simdMode) {
  (void)simdMode;

  if (count == 0) {
    *out = 0.0f;
    return;
  }

  switch (op) {
  case ReduceSum: {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      sum += in[i];
    }
    *out = sum;
    break;
  }
  case ReduceMean: {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      sum += in[i];
    }
    *out = sum / static_cast<float>(count);
    break;
  }
  case ReduceMin: {
    float minVal = in[0];
    for (size_t i = 1; i < count; ++i) {
      if (in[i] < minVal) {
        minVal = in[i];
      }
    }
    *out = minVal;
    break;
  }
  case ReduceMax: {
    float maxVal = in[0];
    for (size_t i = 1; i < count; ++i) {
      if (in[i] > maxVal) {
        maxVal = in[i];
      }
    }
    *out = maxVal;
    break;
  }
  case ReduceProd: {
    float prod = 1.0f;
    for (size_t i = 0; i < count; ++i) {
      prod *= in[i];
    }
    *out = prod;
    break;
  }
  case ReduceAny: {
    float result = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      if (in[i] != 0.0f) {
        result = 1.0f;
        break;
      }
    }
    *out = result;
    break;
  }
  case ReduceAll: {
    float result = 1.0f;
    for (size_t i = 0; i < count; ++i) {
      if (in[i] == 0.0f) {
        result = 0.0f;
        break;
      }
    }
    *out = result;
    break;
  }
  default:
    *out = 0.0f;
    break;
  }
}

void executeMatMulKernel(
    const float *a, const float *b, float *c, size_t M, size_t K, size_t N) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += a[i * K + k] * b[k * N + j];
      }
      c[i * N + j] = sum;
    }
  }
}

void executeTransposeKernel(const float *a, float *b, size_t M, size_t N) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      b[j * M + i] = a[i * N + j];
    }
  }
}

void executeDotKernel(const float *a,
                      const float *b,
                      float *result,
                      size_t count) {
  float sum = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    sum += a[i] * b[i];
  }
  *result = sum;
}

} // namespace cut
