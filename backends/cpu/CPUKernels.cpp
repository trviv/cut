#include "CPUKernels.h"
#include "CPUSIMDConfig.h"
#include "CPUSIMDKernels.h"

#include <algorithm>
#include <cmath>

namespace cut {

SIMDMode getEffectiveSIMDMode(SIMDMode requested) {
  if (requested == SIMDMode::Auto) {
    // Return best available at compile time
#if CUT_SIMD_AVX
    return SIMDMode::AVX;
#elif CUT_SIMD_SSE
    return SIMDMode::SSE;
#else
    return SIMDMode::Scalar;
#endif
  }

  // Check if requested mode is available
  if (requested == SIMDMode::AVX) {
#if CUT_SIMD_AVX
    return SIMDMode::AVX;
#elif CUT_SIMD_SSE
    return SIMDMode::SSE; // Fallback to SSE
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

// Binary operation implementations
namespace {

inline float opAdd(float a, float b) {
  return a + b;
}
inline float opSub(float a, float b) {
  return a - b;
}
inline float opMul(float a, float b) {
  return a * b;
}
inline float opDiv(float a, float b) {
  return a / b;
}
inline float opMod(float a, float b) {
  return a - b * std::floor(a / b);
}
inline float opPow(float a, float b) {
  return std::pow(a, b);
}
inline float opFloorDiv(float a, float b) {
  return std::floor(a / b);
}
inline float opEqual(float a, float b) {
  return a == b ? 1.0f : 0.0f;
}
inline float opNotEqual(float a, float b) {
  return a != b ? 1.0f : 0.0f;
}
inline float opLess(float a, float b) {
  return a < b ? 1.0f : 0.0f;
}
inline float opLessEqual(float a, float b) {
  return a <= b ? 1.0f : 0.0f;
}
inline float opGreater(float a, float b) {
  return a > b ? 1.0f : 0.0f;
}
inline float opGreaterEqual(float a, float b) {
  return a >= b ? 1.0f : 0.0f;
}
inline float opMin(float a, float b) {
  return std::min(a, b);
}
inline float opMax(float a, float b) {
  return std::max(a, b);
}

// Unary operation implementations
inline float opNeg(float x) {
  return -x;
}
inline float opAbs(float x) {
  return std::abs(x);
}
inline float opSqrt(float x) {
  return std::sqrt(x);
}
inline float opExp(float x) {
  return std::exp(x);
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
inline float opSign(float x) {
  return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
}
inline float opReciprocal(float x) {
  return 1.0f / x;
}
inline float opSquare(float x) {
  return x * x;
}

// New unary operations
inline float opExpm1(float x) {
  return std::expm1(x);
}
inline float opLog1p(float x) {
  return std::log1p(x);
}
inline float opCbrt(float x) {
  return std::cbrt(x);
}
inline float opExp2(float x) {
  return std::exp2(x);
}
inline float opDegrees(float x) {
  return x * 180.0f / 3.14159265358979323846f;
}
inline float opRadians(float x) {
  return x * 3.14159265358979323846f / 180.0f;
}
inline float opLogicalNot(float x) {
  return x == 0.0f ? 1.0f : 0.0f;
}
inline float opBitwiseNot(float x) {
  int32_t ix = static_cast<int32_t>(x);
  return static_cast<float>(~ix);
}
inline float opRelu(float x) {
  return x > 0.0f ? x : 0.0f;
}
inline float opSigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}
inline float opGelu(float x) {
  // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  const float sqrt2pi = 0.7978845608028654f; // sqrt(2/pi)
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + std::tanh(sqrt2pi * (x + 0.044715f * x3)));
}
inline float opSilu(float x) {
  // SiLU/Swish: x * sigmoid(x)
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

// New binary operations
inline float opBitwiseAnd(float a, float b) {
  return static_cast<float>(static_cast<int32_t>(a) & static_cast<int32_t>(b));
}
inline float opBitwiseOr(float a, float b) {
  return static_cast<float>(static_cast<int32_t>(a) | static_cast<int32_t>(b));
}
inline float opBitwiseXor(float a, float b) {
  return static_cast<float>(static_cast<int32_t>(a) ^ static_cast<int32_t>(b));
}
inline float opLeftShift(float a, float b) {
  return static_cast<float>(static_cast<int32_t>(a) << static_cast<int32_t>(b));
}
inline float opRightShift(float a, float b) {
  return static_cast<float>(static_cast<int32_t>(a) >> static_cast<int32_t>(b));
}
inline float opLogicalAnd(float a, float b) {
  return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
}
inline float opLogicalOr(float a, float b) {
  return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
}
inline float opLogicalXor(float a, float b) {
  return ((a != 0.0f) != (b != 0.0f)) ? 1.0f : 0.0f;
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

template <typename BinaryOp>
void binaryLoop(const float *a,
                const float *b,
                float *out,
                size_t start,
                size_t end,
                BinaryOp op) {
  for (size_t i = start; i < end; ++i) {
    out[i] = op(a[i], b[i]);
  }
}

template <typename UnaryOp>
void unaryLoop(
    const float *in, float *out, size_t start, size_t end, UnaryOp op) {
  for (size_t i = start; i < end; ++i) {
    out[i] = op(in[i]);
  }
}

template <typename BinaryOp>
void binaryVecScalarLoop(const float *a,
                         float scalar,
                         float *out,
                         size_t start,
                         size_t end,
                         BinaryOp op) {
  for (size_t i = start; i < end; ++i) {
    out[i] = op(a[i], scalar);
  }
}

} // namespace

void executeBinaryKernel(OperatorEnum op,
                         const float *a,
                         const float *b,
                         float *out,
                         size_t start,
                         size_t end,
                         SIMDMode simdMode) {
  const float *aStart = a + start;
  const float *bStart = b + start;
  float *outStart = out + start;
  size_t count = end - start;

  // Get effective SIMD mode based on what's compiled in
  SIMDMode effectiveMode = getEffectiveSIMDMode(simdMode);

  switch (op) {
  case BinaryVecVecAdd:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxAdd(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseAdd(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opAdd);
    break;
  case BinaryVecVecSub:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxSub(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseSub(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opSub);
    break;
  case BinaryVecVecMul:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxMul(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseMul(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opMul);
    break;
  case BinaryVecVecDiv:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxDiv(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseDiv(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opDiv);
    break;
  case BinaryVecVecMod:
    // Modulo doesn't have a direct SIMD instruction, use scalar
    binaryLoop(a, b, out, start, end, opMod);
    break;
  case BinaryVecVecPow:
    // Power doesn't have a direct SIMD instruction, use scalar
    binaryLoop(a, b, out, start, end, opPow);
    break;
  case BinaryVecVecFloorDiv:
    // FloorDiv doesn't have a direct SIMD instruction, use scalar
    binaryLoop(a, b, out, start, end, opFloorDiv);
    break;
  case BinaryVecVecEqual:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxEqual(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseEqual(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opEqual);
    break;
  case BinaryVecVecNotEqual:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxNotEqual(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseNotEqual(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opNotEqual);
    break;
  case BinaryVecVecLess:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxLess(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseLess(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opLess);
    break;
  case BinaryVecVecLessEqual:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxLessEqual(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseLessEqual(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opLessEqual);
    break;
  case BinaryVecVecGreater:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxGreater(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseGreater(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opGreater);
    break;
  case BinaryVecVecGreaterEqual:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxGreaterEqual(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseGreaterEqual(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opGreaterEqual);
    break;
  case BinaryVecVecMin:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxMin(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseMin(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opMin);
    break;
  case BinaryVecVecMax:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxMax(aStart, bStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseMax(aStart, bStart, outStart, count);
      break;
    }
#endif
    binaryLoop(a, b, out, start, end, opMax);
    break;
  // New binary vec-vec operations
  case BinaryVecVecBitwiseAnd:
    binaryLoop(a, b, out, start, end, opBitwiseAnd);
    break;
  case BinaryVecVecBitwiseOr:
    binaryLoop(a, b, out, start, end, opBitwiseOr);
    break;
  case BinaryVecVecBitwiseXor:
    binaryLoop(a, b, out, start, end, opBitwiseXor);
    break;
  case BinaryVecVecLeftShift:
    binaryLoop(a, b, out, start, end, opLeftShift);
    break;
  case BinaryVecVecRightShift:
    binaryLoop(a, b, out, start, end, opRightShift);
    break;
  case BinaryVecVecLogicalAnd:
    binaryLoop(a, b, out, start, end, opLogicalAnd);
    break;
  case BinaryVecVecLogicalOr:
    binaryLoop(a, b, out, start, end, opLogicalOr);
    break;
  case BinaryVecVecLogicalXor:
    binaryLoop(a, b, out, start, end, opLogicalXor);
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
}

void executeBinaryVecScalarKernel(OperatorEnum op,
                                  const float *a,
                                  float scalar,
                                  float *out,
                                  size_t start,
                                  size_t end,
                                  SIMDMode simdMode) {
  (void)simdMode; // Currently using scalar implementations only

  switch (op) {
  case BinaryVecScalarAdd:
    binaryVecScalarLoop(a, scalar, out, start, end, opAdd);
    break;
  case BinaryVecScalarSub:
    binaryVecScalarLoop(a, scalar, out, start, end, opSub);
    break;
  case BinaryVecScalarMul:
    binaryVecScalarLoop(a, scalar, out, start, end, opMul);
    break;
  case BinaryVecScalarDiv:
    binaryVecScalarLoop(a, scalar, out, start, end, opDiv);
    break;
  case BinaryVecScalarMod:
    binaryVecScalarLoop(a, scalar, out, start, end, opMod);
    break;
  case BinaryVecScalarPow:
    binaryVecScalarLoop(a, scalar, out, start, end, opPow);
    break;
  case BinaryVecScalarFloorDiv:
    binaryVecScalarLoop(a, scalar, out, start, end, opFloorDiv);
    break;
  case BinaryVecScalarEqual:
    binaryVecScalarLoop(a, scalar, out, start, end, opEqual);
    break;
  case BinaryVecScalarNotEqual:
    binaryVecScalarLoop(a, scalar, out, start, end, opNotEqual);
    break;
  case BinaryVecScalarLess:
    binaryVecScalarLoop(a, scalar, out, start, end, opLess);
    break;
  case BinaryVecScalarLessEqual:
    binaryVecScalarLoop(a, scalar, out, start, end, opLessEqual);
    break;
  case BinaryVecScalarGreater:
    binaryVecScalarLoop(a, scalar, out, start, end, opGreater);
    break;
  case BinaryVecScalarGreaterEqual:
    binaryVecScalarLoop(a, scalar, out, start, end, opGreaterEqual);
    break;
  case BinaryVecScalarMin:
    binaryVecScalarLoop(a, scalar, out, start, end, opMin);
    break;
  case BinaryVecScalarMax:
    binaryVecScalarLoop(a, scalar, out, start, end, opMax);
    break;
  // New binary vec-scalar operations
  case BinaryVecScalarBitwiseAnd:
    binaryVecScalarLoop(a, scalar, out, start, end, opBitwiseAnd);
    break;
  case BinaryVecScalarBitwiseOr:
    binaryVecScalarLoop(a, scalar, out, start, end, opBitwiseOr);
    break;
  case BinaryVecScalarBitwiseXor:
    binaryVecScalarLoop(a, scalar, out, start, end, opBitwiseXor);
    break;
  case BinaryVecScalarLeftShift:
    binaryVecScalarLoop(a, scalar, out, start, end, opLeftShift);
    break;
  case BinaryVecScalarRightShift:
    binaryVecScalarLoop(a, scalar, out, start, end, opRightShift);
    break;
  case BinaryVecScalarLogicalAnd:
    binaryVecScalarLoop(a, scalar, out, start, end, opLogicalAnd);
    break;
  case BinaryVecScalarLogicalOr:
    binaryVecScalarLoop(a, scalar, out, start, end, opLogicalOr);
    break;
  case BinaryVecScalarLogicalXor:
    binaryVecScalarLoop(a, scalar, out, start, end, opLogicalXor);
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
}

void executeUnaryKernel(OperatorEnum op,
                        const float *in,
                        float *out,
                        size_t start,
                        size_t end,
                        SIMDMode simdMode) {
  const float *inStart = in + start;
  float *outStart = out + start;
  size_t count = end - start;

  // Get effective SIMD mode based on what's compiled in
  SIMDMode effectiveMode = getEffectiveSIMDMode(simdMode);

  switch (op) {
  case UnaryNeg:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxNeg(inStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseNeg(inStart, outStart, count);
      break;
    }
#endif
    unaryLoop(in, out, start, end, opNeg);
    break;
  case UnaryAbs:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxAbs(inStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseAbs(inStart, outStart, count);
      break;
    }
#endif
    unaryLoop(in, out, start, end, opAbs);
    break;
  case UnarySqrt:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxSqrt(inStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseSqrt(inStart, outStart, count);
      break;
    }
#endif
    unaryLoop(in, out, start, end, opSqrt);
    break;
  case UnaryExp:
    // Exp doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opExp);
    break;
  case UnaryLog:
    // Log doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opLog);
    break;
  case UnaryLog2:
    // Log2 doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opLog2);
    break;
  case UnaryLog10:
    // Log10 doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opLog10);
    break;
  case UnarySin:
    // Sin doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opSin);
    break;
  case UnaryCos:
    // Cos doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opCos);
    break;
  case UnaryTan:
    // Tan doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opTan);
    break;
  case UnaryAsin:
    // Asin doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opAsin);
    break;
  case UnaryAcos:
    // Acos doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opAcos);
    break;
  case UnaryAtan:
    // Atan doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opAtan);
    break;
  case UnarySinh:
    // Sinh doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opSinh);
    break;
  case UnaryCosh:
    // Cosh doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opCosh);
    break;
  case UnaryTanh:
    // Tanh doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opTanh);
    break;
  case UnaryFloor:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxFloor(inStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE4_1
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseFloor(inStart, outStart, count);
      break;
    }
#endif
    unaryLoop(in, out, start, end, opFloor);
    break;
  case UnaryCeil:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxCeil(inStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE4_1
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseCeil(inStart, outStart, count);
      break;
    }
#endif
    unaryLoop(in, out, start, end, opCeil);
    break;
  case UnaryRound:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxRound(inStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE4_1
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseRound(inStart, outStart, count);
      break;
    }
#endif
    unaryLoop(in, out, start, end, opRound);
    break;
  case UnarySign:
    // Sign doesn't have a direct SIMD instruction, use scalar
    unaryLoop(in, out, start, end, opSign);
    break;
  case UnaryReciprocal:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxReciprocal(inStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseReciprocal(inStart, outStart, count);
      break;
    }
#endif
    unaryLoop(in, out, start, end, opReciprocal);
    break;
  case UnarySquare:
#if CUT_SIMD_AVX
    if (effectiveMode == SIMDMode::AVX) {
      simd::avxSquare(inStart, outStart, count);
      break;
    }
#endif
#if CUT_SIMD_SSE
    if (effectiveMode == SIMDMode::SSE) {
      simd::sseSquare(inStart, outStart, count);
      break;
    }
#endif
    unaryLoop(in, out, start, end, opSquare);
    break;
  // New unary operations
  case UnaryExpm1:
    unaryLoop(in, out, start, end, opExpm1);
    break;
  case UnaryLog1p:
    unaryLoop(in, out, start, end, opLog1p);
    break;
  case UnaryCbrt:
    unaryLoop(in, out, start, end, opCbrt);
    break;
  case UnaryExp2:
    unaryLoop(in, out, start, end, opExp2);
    break;
  case UnaryDegrees:
    unaryLoop(in, out, start, end, opDegrees);
    break;
  case UnaryRadians:
    unaryLoop(in, out, start, end, opRadians);
    break;
  case UnaryLogicalNot:
    unaryLoop(in, out, start, end, opLogicalNot);
    break;
  case UnaryBitwiseNot:
    unaryLoop(in, out, start, end, opBitwiseNot);
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
}

void executeTernaryClampKernel(const float *in,
                               float minVal,
                               float maxVal,
                               float *out,
                               size_t start,
                               size_t end,
                               SIMDMode simdMode) {
  (void)simdMode; // Currently using scalar implementation

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
  (void)simdMode; // Currently using scalar implementation

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
    // Logical OR: returns 1.0 if any element is non-zero
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
    // Logical AND: returns 1.0 only if all elements are non-zero
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
  // Simple naive matrix multiplication: C[i,j] = sum_k(A[i,k] * B[k,j])
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
  // Transpose: B[j,i] = A[i,j]
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
