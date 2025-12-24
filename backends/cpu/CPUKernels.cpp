#include "CPUKernels.h"

#include <algorithm>
#include <cmath>

namespace cut {

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

void executeBinaryKernel(CPUKernelType kernelType,
                         const float *a,
                         const float *b,
                         float *out,
                         size_t start,
                         size_t end) {
  switch (kernelType) {
  case CPUKernelType::BinaryVecVecAdd:
    binaryLoop(a, b, out, start, end, opAdd);
    break;
  case CPUKernelType::BinaryVecVecSub:
    binaryLoop(a, b, out, start, end, opSub);
    break;
  case CPUKernelType::BinaryVecVecMul:
    binaryLoop(a, b, out, start, end, opMul);
    break;
  case CPUKernelType::BinaryVecVecDiv:
    binaryLoop(a, b, out, start, end, opDiv);
    break;
  case CPUKernelType::BinaryVecVecMod:
    binaryLoop(a, b, out, start, end, opMod);
    break;
  case CPUKernelType::BinaryVecVecPow:
    binaryLoop(a, b, out, start, end, opPow);
    break;
  case CPUKernelType::BinaryVecVecFloorDiv:
    binaryLoop(a, b, out, start, end, opFloorDiv);
    break;
  case CPUKernelType::BinaryVecVecEqual:
    binaryLoop(a, b, out, start, end, opEqual);
    break;
  case CPUKernelType::BinaryVecVecNotEqual:
    binaryLoop(a, b, out, start, end, opNotEqual);
    break;
  case CPUKernelType::BinaryVecVecLess:
    binaryLoop(a, b, out, start, end, opLess);
    break;
  case CPUKernelType::BinaryVecVecLessEqual:
    binaryLoop(a, b, out, start, end, opLessEqual);
    break;
  case CPUKernelType::BinaryVecVecGreater:
    binaryLoop(a, b, out, start, end, opGreater);
    break;
  case CPUKernelType::BinaryVecVecGreaterEqual:
    binaryLoop(a, b, out, start, end, opGreaterEqual);
    break;
  case CPUKernelType::BinaryVecVecMin:
    binaryLoop(a, b, out, start, end, opMin);
    break;
  case CPUKernelType::BinaryVecVecMax:
    binaryLoop(a, b, out, start, end, opMax);
    break;
  default:
    break;
  }
}

void executeUnaryKernel(CPUKernelType kernelType,
                        const float *in,
                        float *out,
                        size_t start,
                        size_t end) {
  switch (kernelType) {
  case CPUKernelType::UnaryNeg:
    unaryLoop(in, out, start, end, opNeg);
    break;
  case CPUKernelType::UnaryAbs:
    unaryLoop(in, out, start, end, opAbs);
    break;
  case CPUKernelType::UnarySqrt:
    unaryLoop(in, out, start, end, opSqrt);
    break;
  case CPUKernelType::UnaryExp:
    unaryLoop(in, out, start, end, opExp);
    break;
  case CPUKernelType::UnaryLog:
    unaryLoop(in, out, start, end, opLog);
    break;
  case CPUKernelType::UnaryLog2:
    unaryLoop(in, out, start, end, opLog2);
    break;
  case CPUKernelType::UnaryLog10:
    unaryLoop(in, out, start, end, opLog10);
    break;
  case CPUKernelType::UnarySin:
    unaryLoop(in, out, start, end, opSin);
    break;
  case CPUKernelType::UnaryCos:
    unaryLoop(in, out, start, end, opCos);
    break;
  case CPUKernelType::UnaryTan:
    unaryLoop(in, out, start, end, opTan);
    break;
  case CPUKernelType::UnaryAsin:
    unaryLoop(in, out, start, end, opAsin);
    break;
  case CPUKernelType::UnaryAcos:
    unaryLoop(in, out, start, end, opAcos);
    break;
  case CPUKernelType::UnaryAtan:
    unaryLoop(in, out, start, end, opAtan);
    break;
  case CPUKernelType::UnarySinh:
    unaryLoop(in, out, start, end, opSinh);
    break;
  case CPUKernelType::UnaryCosh:
    unaryLoop(in, out, start, end, opCosh);
    break;
  case CPUKernelType::UnaryTanh:
    unaryLoop(in, out, start, end, opTanh);
    break;
  case CPUKernelType::UnaryFloor:
    unaryLoop(in, out, start, end, opFloor);
    break;
  case CPUKernelType::UnaryCeil:
    unaryLoop(in, out, start, end, opCeil);
    break;
  case CPUKernelType::UnaryRound:
    unaryLoop(in, out, start, end, opRound);
    break;
  case CPUKernelType::UnarySign:
    unaryLoop(in, out, start, end, opSign);
    break;
  case CPUKernelType::UnaryReciprocal:
    unaryLoop(in, out, start, end, opReciprocal);
    break;
  case CPUKernelType::UnarySquare:
    unaryLoop(in, out, start, end, opSquare);
    break;
  default:
    break;
  }
}

} // namespace cut
