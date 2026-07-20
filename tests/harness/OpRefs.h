#pragma once

#include <ComputeCommon.h>
#include <Operations.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace cut {

// ==== CONSTANTS ====
inline constexpr std::array<uint32_t, 6> kTestDimSizes = {1, 3, 7, 9, 13, 17};
inline constexpr std::array<DataType, 4> kAllDataTypes = {
    DataType::Float32, DataType::Float16, DataType::UInt32, DataType::Int32};
inline constexpr std::array<OperatorEnum, 29> kBinaryVecVecOps = {
    // Arithmetic
    BinaryAdd, BinarySub, BinaryMul, BinaryDiv, BinaryMod, BinaryPow,
    BinaryFloorDiv,
    // Comparison
    BinaryEqual, BinaryNotEqual, BinaryLess, BinaryLessEqual, BinaryGreater,
    BinaryGreaterEqual,
    // Min/Max
    BinaryMin, BinaryMax,
    // Bitwise
    BinaryBitwiseAnd, BinaryBitwiseOr, BinaryBitwiseXor, BinaryLeftShift,
    BinaryRightShift,
    // Logical
    BinaryLogicalAnd, BinaryLogicalOr, BinaryLogicalXor,
    // Math
    BinaryAtan2, BinaryHypot, BinaryCopysign, BinaryFmod,
    // Numerically stable log-sum-exp
    BinaryLogaddexp, BinaryLogaddexp2};
inline constexpr std::array<OperatorEnum, 22> kIntBinaryVecVecOps = {
    // Arithmetic
    BinaryAdd, BinarySub, BinaryMul, BinaryDiv, BinaryMod, BinaryFloorDiv,
    // Comparison
    BinaryEqual, BinaryNotEqual, BinaryLess, BinaryLessEqual, BinaryGreater,
    BinaryGreaterEqual,
    // Min/Max
    BinaryMin, BinaryMax,
    // Bitwise
    BinaryBitwiseAnd, BinaryBitwiseOr, BinaryBitwiseXor, BinaryLeftShift,
    BinaryRightShift,
    // Logical
    BinaryLogicalAnd, BinaryLogicalOr, BinaryLogicalXor};
inline constexpr std::array<OperatorEnum, 33> kBinaryVecScalarOps = {
    // Arithmetic
    BinaryAdd, BinarySub, BinaryMul, BinaryDiv, BinaryMod, BinaryPow,
    BinaryFloorDiv,
    // Comparison
    BinaryEqual, BinaryNotEqual, BinaryLess, BinaryLessEqual, BinaryGreater,
    BinaryGreaterEqual,
    // Min/Max
    BinaryMin, BinaryMax,
    // Bitwise
    BinaryBitwiseAnd, BinaryBitwiseOr, BinaryBitwiseXor, BinaryLeftShift,
    BinaryRightShift,
    // Logical
    BinaryLogicalAnd, BinaryLogicalOr, BinaryLogicalXor,
    // Math
    BinaryAtan2, BinaryHypot, BinaryCopysign, BinaryFmod,
    // Activation
    BinaryLeakyRelu,
    // Parameterized activations
    BinaryPrelu, BinaryHardshrink, BinarySoftshrink,
    // Numerically stable log-sum-exp
    BinaryLogaddexp, BinaryLogaddexp2};
inline constexpr std::array<OperatorEnum, 55> kUnaryOps = {
    // Basic
    UnaryNeg, UnaryAbs, UnarySqrt, UnarySquare, UnaryReciprocal, UnarySign,
    // Exponential/Logarithmic
    UnaryExp, UnaryExp2, UnaryExpm1, UnaryLog, UnaryLog2, UnaryLog10,
    UnaryLog1p,
    // Trigonometric
    UnarySin, UnaryCos, UnaryTan, UnaryAsin, UnaryAcos, UnaryAtan,
    // Hyperbolic
    UnarySinh, UnaryCosh, UnaryTanh,
    // Rounding
    UnaryFloor, UnaryCeil, UnaryRound,
    // Special Math
    UnaryCbrt, UnaryDegrees, UnaryRadians,
    // Logical/Bitwise
    UnaryLogicalNot, UnaryBitwiseNot,
    // Activation Functions
    UnaryRelu, UnarySigmoid, UnaryGelu, UnarySilu, UnarySoftplus,
    // Check Operations
    UnaryIsNan, UnaryIsInf,
    // Extended Activations (Phase 1)
    UnaryRelu6, UnaryElu, UnarySelu, UnaryCelu, UnaryMish, UnaryHardswish,
    UnaryHardsigmoid, UnaryHardtanh, UnarySoftsign, UnaryLogSigmoid,
    UnaryTanhshrink,
    // Extended Math (Phase 2)
    UnaryRsqrt, UnaryTrunc, UnaryFrac, UnaryAsinh, UnaryAcosh, UnaryAtanh,
    UnaryIsFinite};
inline constexpr std::array<OperatorEnum, 7> kDimReductionOps = {
    ReduceSum,  ReduceMean, ReduceMin, ReduceMax,
    ReduceProd, ReduceAny,  ReduceAll};
// ==== HALF UTILS ====
inline uint16_t floatToHalf(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  uint16_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127;
  uint32_t mantissa = bits & 0x7FFFFF;
  if (exponent == 128) { // Inf or NaN
    return sign | 0x7C00 | (mantissa ? (mantissa >> 13) | 1 : 0);
  }
  if (exponent < -14) { // Underflow to zero
    return sign;
  }
  if (exponent > 15) { // Overflow to Inf
    return sign | 0x7C00;
  }
  return sign | ((exponent + 15) << 10) | (mantissa >> 13);
}

inline float halfToFloat(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x03FF;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign; // Zero
    } else {
      // Denormalized
      exponent = 1;
      while (!(mantissa & 0x0400)) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x03FF;
      bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7F800000 | (mantissa << 13); // Inf or NaN
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline std::vector<uint16_t> floatsToHalves(const std::vector<float> &v) {
  std::vector<uint16_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = floatToHalf(v[i]);
  return out;
}

inline std::vector<float> halvesToFloats(const std::vector<uint16_t> &v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = halfToFloat(v[i]);
  return out;
}
inline uint16_t f32_to_f16(float value) {
  uint32_t f32;
  std::memcpy(&f32, &value, sizeof(f32));
  uint32_t sign = (f32 >> 16) & 0x8000;
  int32_t exponent = ((f32 >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = (f32 >> 13) & 0x03FF;
  if (exponent <= 0) {
    return static_cast<uint16_t>(sign); // flush to zero
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00); // infinity
  }
  return static_cast<uint16_t>(sign | (exponent << 10) | mantissa);
}
inline uint8_t packNibbles(uint8_t lo, uint8_t hi) {
  return (lo & 0xFu) | ((hi & 0xFu) << 4);
}
// ==== SHAPE/DATA ====
inline std::vector<std::vector<uint32_t>> generateShapes(size_t numDims) {
  std::vector<std::vector<uint32_t>> shapes;

  if (numDims == 1) {
    // 1D shapes can use any size (no padding between rows)
    for (uint32_t size : kTestDimSizes) {
      shapes.push_back({size});
    }
  } else {
    // Multi-dimensional: innermost dimension must be multiple of 4
    // Outer dimensions can be any size from kTestDimSizes
    constexpr std::array<uint32_t, 4> innerDimSizes = {4, 8, 12, 16};

    for (uint32_t innerSize : innerDimSizes) {
      for (uint32_t outerSize : kTestDimSizes) {
        std::vector<uint32_t> shape(numDims, outerSize);
        shape.back() = innerSize; // Set innermost dimension
        shapes.push_back(shape);
      }
    }

    // Also add some mixed shapes
    if (numDims >= 2) {
      shapes.push_back({3, 8});
      shapes.push_back({7, 4});
      shapes.push_back({9, 12});
    }
    if (numDims >= 3) {
      shapes.push_back({3, 5, 8});
      shapes.push_back({2, 7, 4});
    }
    if (numDims >= 4) {
      shapes.push_back({2, 3, 5, 8});
      shapes.push_back({1, 3, 7, 4});
    }
  }

  return shapes;
}

// Calculate total elements in a shape
inline uint32_t totalElements(const std::vector<uint32_t> &shape) {
  uint32_t total = 1;
  for (auto dim : shape) {
    total *= dim;
  }
  return total;
}

// Format shape as string for test naming
inline std::string shapeToString(const std::vector<uint32_t> &shape) {
  std::string result;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      result += "x";
    result += std::to_string(shape[i]);
  }
  return result;
}
template <typename T>
std::vector<T> generateTestData(size_t count, uint32_t seed = 42) {
  std::mt19937 gen(seed);
  std::vector<T> data(count);

  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(T{0.1}, T{10.0});
    for (auto &v : data) {
      v = dist(gen);
    }
  } else if constexpr (std::is_signed_v<T>) {
    std::uniform_int_distribution<T> dist(T{1}, T{100});
    for (auto &v : data) {
      v = dist(gen);
    }
  } else {
    std::uniform_int_distribution<T> dist(T{1}, T{100});
    for (auto &v : data) {
      v = dist(gen);
    }
  }

  return data;
}
// ==== REFS ====
// Binary vec-vec reference
template <typename T>
T binaryVecVecRef(OperatorEnum op, T a, T b) {
  switch (op) {
  case BinaryAdd:
    return a + b;
  case BinarySub:
    return a - b;
  case BinaryMul:
    return a * b;
  case BinaryDiv:
    return a / b;
  case BinaryMod:
    if constexpr (std::is_floating_point_v<T>) {
      return std::fmod(a, b);
    } else {
      return a % b;
    }
  case BinaryPow:
    if constexpr (std::is_floating_point_v<T>) {
      return std::pow(a, b);
    } else {
      return static_cast<T>(
          std::pow(static_cast<double>(a), static_cast<double>(b)));
    }
  case BinaryFloorDiv:
    if constexpr (std::is_floating_point_v<T>) {
      return std::floor(a / b);
    } else {
      return a / b;
    }
  case BinaryEqual:
    return static_cast<T>(a == b ? 1 : 0);
  case BinaryNotEqual:
    return static_cast<T>(a != b ? 1 : 0);
  case BinaryLess:
    return static_cast<T>(a < b ? 1 : 0);
  case BinaryLessEqual:
    return static_cast<T>(a <= b ? 1 : 0);
  case BinaryGreater:
    return static_cast<T>(a > b ? 1 : 0);
  case BinaryGreaterEqual:
    return static_cast<T>(a >= b ? 1 : 0);
  case BinaryMin:
    return std::min(a, b);
  case BinaryMax:
    return std::max(a, b);
  // Bitwise operations - for floats, match GPU shader behavior:
  // intBitsToFloat(floatBitsToInt(a) OP floatBitsToInt(b))
  case BinaryBitwiseAnd:
    if constexpr (std::is_integral_v<T>) {
      return a & b;
    } else {
      int ia, ib, ir;
      memcpy(&ia, &a, sizeof(int));
      memcpy(&ib, &b, sizeof(int));
      ir = ia & ib;
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  case BinaryBitwiseOr:
    if constexpr (std::is_integral_v<T>) {
      return a | b;
    } else {
      int ia, ib, ir;
      memcpy(&ia, &a, sizeof(int));
      memcpy(&ib, &b, sizeof(int));
      ir = ia | ib;
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  case BinaryBitwiseXor:
    if constexpr (std::is_integral_v<T>) {
      return a ^ b;
    } else {
      int ia, ib, ir;
      memcpy(&ia, &a, sizeof(int));
      memcpy(&ib, &b, sizeof(int));
      ir = ia ^ ib;
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  case BinaryLeftShift:
    if constexpr (std::is_integral_v<T>) {
      return a << static_cast<int>(b);
    } else {
      uint32_t ua, ub, ur;
      memcpy(&ua, &a, sizeof(uint32_t));
      memcpy(&ub, &b, sizeof(uint32_t));
      ur = ua << (ub & 31u); // SPIR-V masks shift to 5 bits
      T r;
      memcpy(&r, &ur, sizeof(T));
      return r;
    }
  case BinaryRightShift:
    if constexpr (std::is_integral_v<T>) {
      return a >> static_cast<int>(b);
    } else {
      uint32_t ua, ub, ur;
      memcpy(&ua, &a, sizeof(uint32_t));
      memcpy(&ub, &b, sizeof(uint32_t));
      ur = ua >> (ub & 31u); // SPIR-V masks shift to 5 bits
      T r;
      memcpy(&r, &ur, sizeof(T));
      return r;
    }
  // Logical operations
  case BinaryLogicalAnd:
    return static_cast<T>((a != T{0} && b != T{0}) ? 1 : 0);
  case BinaryLogicalOr:
    return static_cast<T>((a != T{0} || b != T{0}) ? 1 : 0);
  case BinaryLogicalXor:
    return static_cast<T>((a != T{0}) != (b != T{0}) ? 1 : 0);
  // Math operations
  case BinaryAtan2:
    if constexpr (std::is_floating_point_v<T>) {
      return std::atan2(a, b);
    } else {
      return static_cast<T>(
          std::atan2(static_cast<double>(a), static_cast<double>(b)));
    }
  case BinaryHypot:
    if constexpr (std::is_floating_point_v<T>) {
      return std::hypot(a, b);
    } else {
      return static_cast<T>(
          std::hypot(static_cast<double>(a), static_cast<double>(b)));
    }
  case BinaryCopysign:
    if constexpr (std::is_floating_point_v<T>) {
      return std::copysign(a, b);
    } else if constexpr (std::is_signed_v<T>) {
      return b >= T{0} ? std::abs(a) : -std::abs(a);
    } else {
      return a; // unsigned values are always non-negative
    }
  case BinaryFmod:
    if constexpr (std::is_floating_point_v<T>) {
      return std::fmod(a, b);
    } else {
      return a % b;
    }
  case BinaryLogaddexp:
    if constexpr (std::is_floating_point_v<T>) {
      return std::max(a, b) + std::log(T{1} + std::exp(-std::abs(a - b)));
    } else {
      double da = static_cast<double>(a), db = static_cast<double>(b);
      return static_cast<T>(std::max(da, db) +
                            std::log(1.0 + std::exp(-std::abs(da - db))));
    }
  case BinaryLogaddexp2:
    if constexpr (std::is_floating_point_v<T>) {
      return std::max(a, b) + std::log2(T{1} + std::exp2(-std::abs(a - b)));
    } else {
      double da = static_cast<double>(a), db = static_cast<double>(b);
      return static_cast<T>(std::max(da, db) +
                            std::log2(1.0 + std::exp2(-std::abs(da - db))));
    }
  default:
    return T{};
  }
}
// Binary vec-scalar reference
template <typename T>
T binaryVecScalarRef(OperatorEnum op, T a, T scalar) {
  // Map vec-scalar ops to corresponding vec-vec logic
  switch (op) {
  case BinaryAdd:
    return a + scalar;
  case BinarySub:
    return a - scalar;
  case BinaryMul:
    return a * scalar;
  case BinaryDiv:
    return a / scalar;
  case BinaryMod:
    if constexpr (std::is_floating_point_v<T>) {
      return std::fmod(a, scalar);
    } else {
      return a % scalar;
    }
  case BinaryPow:
    if constexpr (std::is_floating_point_v<T>) {
      return std::pow(a, scalar);
    } else {
      return static_cast<T>(
          std::pow(static_cast<double>(a), static_cast<double>(scalar)));
    }
  case BinaryFloorDiv:
    if constexpr (std::is_floating_point_v<T>) {
      return std::floor(a / scalar);
    } else {
      return a / scalar;
    }
  case BinaryEqual:
    return static_cast<T>(a == scalar ? 1 : 0);
  case BinaryNotEqual:
    return static_cast<T>(a != scalar ? 1 : 0);
  case BinaryLess:
    return static_cast<T>(a < scalar ? 1 : 0);
  case BinaryLessEqual:
    return static_cast<T>(a <= scalar ? 1 : 0);
  case BinaryGreater:
    return static_cast<T>(a > scalar ? 1 : 0);
  case BinaryGreaterEqual:
    return static_cast<T>(a >= scalar ? 1 : 0);
  case BinaryMin:
    return std::min(a, scalar);
  case BinaryMax:
    return std::max(a, scalar);
  // Bitwise operations — GPU shader uses floatBitsToInt (bit reinterpretation)
  // for both the vector element and the scalar
  case BinaryBitwiseAnd:
    if constexpr (std::is_integral_v<T>) {
      return a & static_cast<T>(scalar);
    } else {
      int ia, is, ir;
      memcpy(&ia, &a, sizeof(int));
      memcpy(&is, &scalar, sizeof(int));
      ir = ia & is;
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  case BinaryBitwiseOr:
    if constexpr (std::is_integral_v<T>) {
      return a | static_cast<T>(scalar);
    } else {
      int ia, is, ir;
      memcpy(&ia, &a, sizeof(int));
      memcpy(&is, &scalar, sizeof(int));
      ir = ia | is;
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  case BinaryBitwiseXor:
    if constexpr (std::is_integral_v<T>) {
      return a ^ static_cast<T>(scalar);
    } else {
      int ia, is, ir;
      memcpy(&ia, &a, sizeof(int));
      memcpy(&is, &scalar, sizeof(int));
      ir = ia ^ is;
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  case BinaryLeftShift:
    if constexpr (std::is_integral_v<T>) {
      return a << static_cast<int>(scalar);
    } else {
      int ia, is;
      memcpy(&ia, &a, sizeof(int));
      memcpy(&is, &scalar, sizeof(int));
      int ir = ia << (static_cast<uint32_t>(is) & 31u);
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  case BinaryRightShift:
    if constexpr (std::is_integral_v<T>) {
      return a >> static_cast<int>(scalar);
    } else {
      int ia, is;
      memcpy(&ia, &a, sizeof(int));
      memcpy(&is, &scalar, sizeof(int));
      int ir = ia >> (static_cast<uint32_t>(is) & 31u);
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  // Logical operations
  case BinaryLogicalAnd:
    return static_cast<T>((a != T{0} && scalar != T{0}) ? 1 : 0);
  case BinaryLogicalOr:
    return static_cast<T>((a != T{0} || scalar != T{0}) ? 1 : 0);
  case BinaryLogicalXor:
    return static_cast<T>((a != T{0}) != (scalar != T{0}) ? 1 : 0);
  // Math operations
  case BinaryAtan2:
    if constexpr (std::is_floating_point_v<T>) {
      return std::atan2(a, scalar);
    } else {
      return static_cast<T>(
          std::atan2(static_cast<double>(a), static_cast<double>(scalar)));
    }
  case BinaryHypot:
    if constexpr (std::is_floating_point_v<T>) {
      return std::hypot(a, scalar);
    } else {
      return static_cast<T>(
          std::hypot(static_cast<double>(a), static_cast<double>(scalar)));
    }
  case BinaryCopysign:
    if constexpr (std::is_floating_point_v<T>) {
      return std::copysign(a, scalar);
    } else if constexpr (std::is_unsigned_v<T>) {
      return a; // unsigned values are always non-negative
    } else {
      return scalar >= T{0} ? std::abs(a) : -std::abs(a);
    }
  case BinaryFmod:
    if constexpr (std::is_floating_point_v<T>) {
      return std::fmod(a, scalar);
    } else {
      return a % static_cast<T>(scalar);
    }
  // Activation
  case BinaryLeakyRelu:
    return a > T{0} ? a : a * scalar;
  // Parameterized activations
  case BinaryPrelu:
    return a >= T{0} ? a : scalar * a;
  case BinaryHardshrink:
    if constexpr (std::is_floating_point_v<T>) {
      return std::abs(a) > scalar ? a : T{0};
    } else {
      return std::abs(static_cast<double>(a)) > static_cast<double>(scalar)
                 ? a
                 : T{0};
    }
  case BinarySoftshrink:
    if constexpr (std::is_floating_point_v<T>) {
      if (a > scalar)
        return a - scalar;
      if (a < -scalar)
        return a + scalar;
      return T{0};
    } else {
      double da = static_cast<double>(a), ds = static_cast<double>(scalar);
      if (da > ds)
        return static_cast<T>(da - ds);
      if (da < -ds)
        return static_cast<T>(da + ds);
      return T{0};
    }
  // Log-sum-exp
  case BinaryLogaddexp:
    if constexpr (std::is_floating_point_v<T>) {
      return std::max(a, scalar) +
             std::log(T{1} + std::exp(-std::abs(a - scalar)));
    } else {
      double da = static_cast<double>(a), ds = static_cast<double>(scalar);
      return static_cast<T>(std::max(da, ds) +
                            std::log(1.0 + std::exp(-std::abs(da - ds))));
    }
  case BinaryLogaddexp2:
    if constexpr (std::is_floating_point_v<T>) {
      return std::max(a, scalar) +
             std::log2(T{1} + std::exp2(-std::abs(a - scalar)));
    } else {
      double da = static_cast<double>(a), ds = static_cast<double>(scalar);
      return static_cast<T>(std::max(da, ds) +
                            std::log2(1.0 + std::exp2(-std::abs(da - ds))));
    }
  default:
    return T{};
  }
}
// Unary reference
template <typename T>
T unaryRef(OperatorEnum op, T a) {
  switch (op) {
  case UnaryNeg:
    return -a;
  case UnaryAbs:
    if constexpr (std::is_unsigned_v<T>) {
      return a;
    } else {
      return std::abs(a);
    }
  case UnarySqrt:
    if constexpr (std::is_floating_point_v<T>) {
      return std::sqrt(a);
    } else {
      return static_cast<T>(std::sqrt(static_cast<double>(a)));
    }
  case UnaryExp:
    if constexpr (std::is_floating_point_v<T>) {
      return std::exp(a);
    } else {
      return static_cast<T>(std::exp(static_cast<double>(a)));
    }
  case UnaryLog:
    if constexpr (std::is_floating_point_v<T>) {
      return std::log(a);
    } else {
      return static_cast<T>(std::log(static_cast<double>(a)));
    }
  case UnaryLog2:
    if constexpr (std::is_floating_point_v<T>) {
      return std::log2(a);
    } else {
      return static_cast<T>(std::log2(static_cast<double>(a)));
    }
  case UnaryLog10:
    if constexpr (std::is_floating_point_v<T>) {
      return std::log10(a);
    } else {
      return static_cast<T>(std::log10(static_cast<double>(a)));
    }
  case UnarySin:
    if constexpr (std::is_floating_point_v<T>) {
      return std::sin(a);
    } else {
      return static_cast<T>(std::sin(static_cast<double>(a)));
    }
  case UnaryCos:
    if constexpr (std::is_floating_point_v<T>) {
      return std::cos(a);
    } else {
      return static_cast<T>(std::cos(static_cast<double>(a)));
    }
  case UnaryTan:
    if constexpr (std::is_floating_point_v<T>) {
      return std::tan(a);
    } else {
      return static_cast<T>(std::tan(static_cast<double>(a)));
    }
  case UnaryAsin:
    if constexpr (std::is_floating_point_v<T>) {
      return std::asin(a);
    } else {
      return static_cast<T>(std::asin(static_cast<double>(a)));
    }
  case UnaryAcos:
    if constexpr (std::is_floating_point_v<T>) {
      return std::acos(a);
    } else {
      return static_cast<T>(std::acos(static_cast<double>(a)));
    }
  case UnaryAtan:
    if constexpr (std::is_floating_point_v<T>) {
      return std::atan(a);
    } else {
      return static_cast<T>(std::atan(static_cast<double>(a)));
    }
  case UnarySinh:
    if constexpr (std::is_floating_point_v<T>) {
      return std::sinh(a);
    } else {
      return static_cast<T>(std::sinh(static_cast<double>(a)));
    }
  case UnaryCosh:
    if constexpr (std::is_floating_point_v<T>) {
      return std::cosh(a);
    } else {
      return static_cast<T>(std::cosh(static_cast<double>(a)));
    }
  case UnaryTanh:
    if constexpr (std::is_floating_point_v<T>) {
      return std::tanh(a);
    } else {
      return static_cast<T>(std::tanh(static_cast<double>(a)));
    }
  case UnaryFloor:
    if constexpr (std::is_floating_point_v<T>) {
      return std::floor(a);
    } else {
      return a;
    }
  case UnaryCeil:
    if constexpr (std::is_floating_point_v<T>) {
      return std::ceil(a);
    } else {
      return a;
    }
  case UnaryRound:
    if constexpr (std::is_floating_point_v<T>) {
      return std::round(a);
    } else {
      return a;
    }
  case UnarySign:
    if constexpr (std::is_unsigned_v<T>) {
      return a > 0 ? T{1} : T{0};
    } else if constexpr (std::is_floating_point_v<T>) {
      return a > 0 ? T{1} : (a < 0 ? T{-1} : T{0});
    } else {
      return a > 0 ? T{1} : (a < 0 ? T{-1} : T{0});
    }
  case UnaryReciprocal:
    if constexpr (std::is_floating_point_v<T>) {
      return T{1} / a;
    } else {
      return a != 0 ? T{1} / a : T{0};
    }
  case UnarySquare:
    return a * a;
  // Additional unary operators
  case UnaryExp2:
    if constexpr (std::is_floating_point_v<T>) {
      return std::exp2(a);
    } else {
      return static_cast<T>(std::exp2(static_cast<double>(a)));
    }
  case UnaryExpm1:
    if constexpr (std::is_floating_point_v<T>) {
      return std::expm1(a);
    } else {
      return static_cast<T>(std::expm1(static_cast<double>(a)));
    }
  case UnaryLog1p:
    if constexpr (std::is_floating_point_v<T>) {
      return std::log1p(a);
    } else {
      return static_cast<T>(std::log1p(static_cast<double>(a)));
    }
  case UnaryCbrt:
    if constexpr (std::is_floating_point_v<T>) {
      return std::cbrt(a);
    } else {
      return static_cast<T>(std::cbrt(static_cast<double>(a)));
    }
  case UnaryDegrees:
    if constexpr (std::is_floating_point_v<T>) {
      return a * T{180.0} / T{M_PI};
    } else {
      return static_cast<T>(static_cast<double>(a) * 180.0 / M_PI);
    }
  case UnaryRadians:
    if constexpr (std::is_floating_point_v<T>) {
      return a * T{M_PI} / T{180.0};
    } else {
      return static_cast<T>(static_cast<double>(a) * M_PI / 180.0);
    }
  case UnaryLogicalNot:
    return static_cast<T>(a == T{0} ? 1 : 0);
  case UnaryBitwiseNot:
    if constexpr (std::is_integral_v<T>) {
      return ~a;
    } else {
      int ia;
      memcpy(&ia, &a, sizeof(int));
      ia = ~ia;
      T r;
      memcpy(&r, &ia, sizeof(T));
      return r;
    }
  case UnaryRelu:
    return a > T{0} ? a : T{0};
  case UnarySigmoid:
    if constexpr (std::is_floating_point_v<T>) {
      return T{1} / (T{1} + std::exp(-a));
    } else {
      return static_cast<T>(1.0 / (1.0 + std::exp(-static_cast<double>(a))));
    }
  case UnaryGelu: {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 *
    // x^3)))
    if constexpr (std::is_floating_point_v<T>) {
      const T sqrt2pi = T{0.7978845608028654}; // sqrt(2/pi)
      const T coeff = T{0.044715};
      return T{0.5} * a * (T{1} + std::tanh(sqrt2pi * (a + coeff * a * a * a)));
    } else {
      double x = static_cast<double>(a);
      const double sqrt2pi = 0.7978845608028654;
      const double coeff = 0.044715;
      return static_cast<T>(
          0.5 * x * (1.0 + std::tanh(sqrt2pi * (x + coeff * x * x * x))));
    }
  }
  case UnarySilu:
    // SiLU/Swish: x * sigmoid(x)
    if constexpr (std::is_floating_point_v<T>) {
      return a / (T{1} + std::exp(-a));
    } else {
      double x = static_cast<double>(a);
      return static_cast<T>(x / (1.0 + std::exp(-x)));
    }
  case UnarySoftplus:
    // Softplus: log(1 + exp(x))
    if constexpr (std::is_floating_point_v<T>) {
      return std::log(T{1} + std::exp(a));
    } else {
      return static_cast<T>(std::log(1.0 + std::exp(static_cast<double>(a))));
    }
  case UnaryIsNan:
    if constexpr (std::is_floating_point_v<T>) {
      return static_cast<T>(std::isnan(a) ? 1 : 0);
    } else {
      return T{0}; // Integers are never NaN
    }
  case UnaryIsInf:
    if constexpr (std::is_floating_point_v<T>) {
      return static_cast<T>(std::isinf(a) ? 1 : 0);
    } else {
      return T{0}; // Integers are never Inf
    }
  // Extended Activations (Phase 1)
  case UnaryRelu6:
    if constexpr (std::is_floating_point_v<T>) {
      return std::clamp(a, T{0}, T{6});
    } else {
      return std::clamp(a, T{0}, T{6});
    }
  case UnaryElu:
    if constexpr (std::is_floating_point_v<T>) {
      return a >= T{0} ? a : std::exp(a) - T{1};
    } else {
      double x = static_cast<double>(a);
      return static_cast<T>(x >= 0 ? x : std::exp(x) - 1.0);
    }
  case UnarySelu: {
    constexpr double kAlpha = 1.6732632423543772;
    constexpr double kScale = 1.0507009873554805;
    double x = static_cast<double>(a);
    return static_cast<T>(kScale * (x >= 0 ? x : kAlpha * (std::exp(x) - 1.0)));
  }
  case UnaryCelu:
    if constexpr (std::is_floating_point_v<T>) {
      return std::max(a, T{0}) + std::min(T{0}, std::exp(a) - T{1});
    } else {
      double x = static_cast<double>(a);
      return static_cast<T>(std::max(x, 0.0) +
                            std::min(0.0, std::exp(x) - 1.0));
    }
  case UnaryMish:
    if constexpr (std::is_floating_point_v<T>) {
      return a * std::tanh(std::log(T{1} + std::exp(a)));
    } else {
      double x = static_cast<double>(a);
      return static_cast<T>(x * std::tanh(std::log(1.0 + std::exp(x))));
    }
  case UnaryHardswish:
    if constexpr (std::is_floating_point_v<T>) {
      return a * std::clamp(a + T{3}, T{0}, T{6}) / T{6};
    } else {
      double x = static_cast<double>(a);
      return static_cast<T>(x * std::clamp(x + 3.0, 0.0, 6.0) / 6.0);
    }
  case UnaryHardsigmoid:
    if constexpr (std::is_floating_point_v<T>) {
      return std::clamp(a / T{6} + T{0.5}, T{0}, T{1});
    } else {
      double x = static_cast<double>(a);
      return static_cast<T>(std::clamp(x / 6.0 + 0.5, 0.0, 1.0));
    }
  case UnaryHardtanh:
    if constexpr (std::is_floating_point_v<T>) {
      return std::clamp(a, T{-1}, T{1});
    } else if constexpr (std::is_unsigned_v<T>) {
      return std::clamp(a, T{0}, T{1});
    } else {
      return std::clamp(a, T{-1}, T{1});
    }
  case UnarySoftsign:
    if constexpr (std::is_floating_point_v<T>) {
      return a / (T{1} + std::abs(a));
    } else {
      double x = static_cast<double>(a);
      return static_cast<T>(x / (1.0 + std::abs(x)));
    }
  case UnaryLogSigmoid:
    if constexpr (std::is_floating_point_v<T>) {
      return -std::log(T{1} + std::exp(-a));
    } else {
      double x = static_cast<double>(a);
      return static_cast<T>(-std::log(1.0 + std::exp(-x)));
    }
  case UnaryTanhshrink:
    if constexpr (std::is_floating_point_v<T>) {
      return a - std::tanh(a);
    } else {
      double x = static_cast<double>(a);
      return static_cast<T>(x - std::tanh(x));
    }
  // Extended Math (Phase 2)
  case UnaryRsqrt:
    if constexpr (std::is_floating_point_v<T>) {
      return T{1} / std::sqrt(a);
    } else {
      return static_cast<T>(1.0 / std::sqrt(static_cast<double>(a)));
    }
  case UnaryTrunc:
    if constexpr (std::is_floating_point_v<T>) {
      return std::trunc(a);
    } else {
      return a;
    }
  case UnaryFrac:
    if constexpr (std::is_floating_point_v<T>) {
      return a - std::floor(a);
    } else {
      return T{0};
    }
  case UnaryAsinh:
    if constexpr (std::is_floating_point_v<T>) {
      return std::asinh(a);
    } else {
      return static_cast<T>(std::asinh(static_cast<double>(a)));
    }
  case UnaryAcosh:
    if constexpr (std::is_floating_point_v<T>) {
      return std::acosh(a);
    } else {
      return static_cast<T>(std::acosh(static_cast<double>(a)));
    }
  case UnaryAtanh:
    if constexpr (std::is_floating_point_v<T>) {
      return std::atanh(a);
    } else {
      return static_cast<T>(std::atanh(static_cast<double>(a)));
    }
  case UnaryIsFinite:
    if constexpr (std::is_floating_point_v<T>) {
      return static_cast<T>(std::isfinite(a) ? 1 : 0);
    } else {
      return T{1}; // Integers are always finite
    }
  default:
    return T{};
  }
}
// Ternary clamp reference
template <typename T>
T ternaryClampRef(T a, T minVal, T maxVal) {
  return std::clamp(a, minVal, maxVal);
}

// Ternary select reference (condition ? x : y)
template <typename T>
T ternarySelectRef(T cond, T x, T y) {
  return (cond != T{0}) ? x : y;
}
// Reduction references
template <typename T>
T reduceRef(OperatorEnum op, const std::vector<T> &data) {
  if (data.empty())
    return T{0};

  switch (op) {
  case ReduceSum: {
    T sum = T{0};
    for (const auto &v : data)
      sum += v;
    return sum;
  }
  case ReduceMean: {
    T sum = T{0};
    for (const auto &v : data)
      sum += v;
    if constexpr (std::is_floating_point_v<T>) {
      return sum / static_cast<T>(data.size());
    } else {
      return sum / static_cast<T>(data.size());
    }
  }
  case ReduceMin: {
    T minVal = data[0];
    for (const auto &v : data)
      if (v < minVal)
        minVal = v;
    return minVal;
  }
  case ReduceMax: {
    T maxVal = data[0];
    for (const auto &v : data)
      if (v > maxVal)
        maxVal = v;
    return maxVal;
  }
  case ReduceProd: {
    T prod = T{1};
    for (const auto &v : data)
      prod *= v;
    return prod;
  }
  case ReduceAny: {
    for (const auto &v : data)
      if (v != T{0})
        return T{1};
    return T{0};
  }
  case ReduceAll: {
    for (const auto &v : data)
      if (v == T{0})
        return T{0};
    return T{1};
  }
  default:
    return T{0};
  }
}
// CPU reference for dimension-wise reduction.
// Reduces data of shape (outerSize, reduceSize, innerSize) along the middle
// dimension, producing outerSize * innerSize outputs.
inline std::vector<float> dimReduceRef(OperatorEnum op,
                                const std::vector<float> &data,
                                uint32_t outerSize,
                                uint32_t reduceSize,
                                uint32_t innerSize) {
  std::vector<float> output(outerSize * innerSize);
  for (uint32_t o = 0; o < outerSize; ++o) {
    for (uint32_t i = 0; i < innerSize; ++i) {
      float val;
      switch (op) {
      case ReduceMin:
        val = std::numeric_limits<float>::max();
        break;
      case ReduceMax:
        val = std::numeric_limits<float>::lowest();
        break;
      case ReduceProd:
      case ReduceAll:
        val = 1.0f;
        break;
      default:
        val = 0.0f;
        break;
      }
      for (uint32_t r = 0; r < reduceSize; ++r) {
        float elem = data[o * reduceSize * innerSize + r * innerSize + i];
        switch (op) {
        case ReduceSum:
        case ReduceMean:
          val += elem;
          break;
        case ReduceMin:
          val = std::min(val, elem);
          break;
        case ReduceMax:
          val = std::max(val, elem);
          break;
        case ReduceProd:
          val *= elem;
          break;
        case ReduceAny:
          val = (val != 0.0f || elem != 0.0f) ? 1.0f : 0.0f;
          break;
        case ReduceAll:
          val = (val != 0.0f && elem != 0.0f) ? 1.0f : 0.0f;
          break;
        default:
          break;
        }
      }
      if (op == ReduceMean) {
        val /= static_cast<float>(reduceSize);
      }
      output[o * innerSize + i] = val;
    }
  }
  return output;
}

// CPU reference for dimension-wise L2 norm.
inline std::vector<float> normDimRef(const std::vector<float> &data,
                              uint32_t outerSize,
                              uint32_t reduceSize,
                              uint32_t innerSize) {
  std::vector<float> output(outerSize * innerSize);
  for (uint32_t o = 0; o < outerSize; ++o) {
    for (uint32_t i = 0; i < innerSize; ++i) {
      float sumSq = 0.0f;
      for (uint32_t r = 0; r < reduceSize; ++r) {
        float elem = data[o * reduceSize * innerSize + r * innerSize + i];
        sumSq += elem * elem;
      }
      output[o * innerSize + i] = std::sqrt(sumSq);
    }
  }
  return output;
}

} // namespace cut
