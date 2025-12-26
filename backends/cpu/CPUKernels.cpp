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
  default:
    break;
  }
}

} // namespace cut
