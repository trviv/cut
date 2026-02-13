#include <gtest/gtest.h>

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cut {
namespace {

// ============================================================================
// Test Configuration
// ============================================================================

// All dimension counts to test (1D, 2D, 3D, 4D)
constexpr std::array<size_t, 4> kDimensionCounts = {1, 2, 3, 4};

// Dimension sizes to test (1-17)
constexpr uint32_t kMinDimSize = 1;
constexpr uint32_t kMaxDimSize = 17;

// Representative dimension sizes for comprehensive testing
// Include odd values: 1, 3, 7, 9, 13, 17
constexpr std::array<uint32_t, 6> kTestDimSizes = {1, 3, 7, 9, 13, 17};

// All data types
constexpr std::array<DataType, 4> kAllDataTypes = {
    DataType::Float32, DataType::Float16, DataType::UInt32, DataType::Int32};

// All binary vec-vec operators
constexpr std::array<OperatorEnum, 29> kBinaryVecVecOps = {
    // Arithmetic
    BinaryVecVecAdd, BinaryVecVecSub, BinaryVecVecMul, BinaryVecVecDiv,
    BinaryVecVecMod, BinaryVecVecPow, BinaryVecVecFloorDiv,
    // Comparison
    BinaryVecVecEqual, BinaryVecVecNotEqual, BinaryVecVecLess,
    BinaryVecVecLessEqual, BinaryVecVecGreater, BinaryVecVecGreaterEqual,
    // Min/Max
    BinaryVecVecMin, BinaryVecVecMax,
    // Bitwise
    BinaryVecVecBitwiseAnd, BinaryVecVecBitwiseOr, BinaryVecVecBitwiseXor,
    BinaryVecVecLeftShift, BinaryVecVecRightShift,
    // Logical
    BinaryVecVecLogicalAnd, BinaryVecVecLogicalOr, BinaryVecVecLogicalXor,
    // Math
    BinaryVecVecAtan2, BinaryVecVecHypot, BinaryVecVecCopysign,
    BinaryVecVecFmod,
    // Numerically stable log-sum-exp
    BinaryVecVecLogaddexp, BinaryVecVecLogaddexp2};

// All binary vec-scalar operators
constexpr std::array<OperatorEnum, 34> kBinaryVecScalarOps = {
    // Arithmetic
    BinaryVecScalarAdd, BinaryVecScalarSub, BinaryVecScalarMul,
    BinaryVecScalarDiv, BinaryVecScalarMod, BinaryVecScalarPow,
    BinaryVecScalarFloorDiv,
    // Comparison
    BinaryVecScalarEqual, BinaryVecScalarNotEqual, BinaryVecScalarLess,
    BinaryVecScalarLessEqual, BinaryVecScalarGreater,
    BinaryVecScalarGreaterEqual,
    // Min/Max
    BinaryVecScalarMin, BinaryVecScalarMax,
    // Bitwise
    BinaryVecScalarBitwiseAnd, BinaryVecScalarBitwiseOr,
    BinaryVecScalarBitwiseXor, BinaryVecScalarLeftShift,
    BinaryVecScalarRightShift,
    // Logical
    BinaryVecScalarLogicalAnd, BinaryVecScalarLogicalOr,
    BinaryVecScalarLogicalXor,
    // Math
    BinaryVecScalarAtan2, BinaryVecScalarHypot, BinaryVecScalarCopysign,
    BinaryVecScalarFmod,
    // Activation
    BinaryVecScalarLeakyRelu,
    // Parameterized activations
    BinaryVecScalarPrelu, BinaryVecScalarHardshrink, BinaryVecScalarSoftshrink,
    // Numerically stable log-sum-exp
    BinaryVecScalarLogaddexp, BinaryVecScalarLogaddexp2};

// All unary operators
constexpr std::array<OperatorEnum, 55> kUnaryOps = {
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

// Ternary operators
constexpr std::array<OperatorEnum, 2> kTernaryOps = {TernaryClamp,
                                                     TernarySelect};

// Reduction operators
constexpr std::array<OperatorEnum, 7> kReductionOps = {
    ReduceSum,  ReduceMean, ReduceMin, ReduceMax,
    ReduceProd, ReduceAny,  ReduceAll};

// Matrix operators
constexpr std::array<OperatorEnum, 3> kMatrixOps = {MatMul, Transpose, Dot};

// ============================================================================
// Helper Functions
// ============================================================================

inline const char *dataTypeName(DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return "Float32";
  case DataType::Float16:
    return "Float16";
  case DataType::UInt32:
    return "UInt32";
  case DataType::Int32:
    return "Int32";
  }
  return "Unknown";
}

inline const char *backendName(BackendType backend) {
  return backend == BackendType::Vulkan ? "Vulkan" : "Unknown";
}

// Helper to check if a binary vec-vec operator has Vulkan shader support
// (New operators not yet implemented in shaders)
inline bool hasVulkanShaderSupport(OperatorEnum op) {
  switch (op) {
  // Original operators with shader support
  case BinaryVecVecAdd:
  case BinaryVecVecSub:
  case BinaryVecVecMul:
  case BinaryVecVecDiv:
  case BinaryVecVecMod:
  case BinaryVecVecPow:
  case BinaryVecVecFloorDiv:
  case BinaryVecVecEqual:
  case BinaryVecVecNotEqual:
  case BinaryVecVecLess:
  case BinaryVecVecLessEqual:
  case BinaryVecVecGreater:
  case BinaryVecVecGreaterEqual:
  case BinaryVecVecMin:
  case BinaryVecVecMax:
  case BinaryVecScalarAdd:
  case BinaryVecScalarSub:
  case BinaryVecScalarMul:
  case BinaryVecScalarDiv:
  case BinaryVecScalarMod:
  case BinaryVecScalarPow:
  case BinaryVecScalarFloorDiv:
  case BinaryVecScalarEqual:
  case BinaryVecScalarNotEqual:
  case BinaryVecScalarLess:
  case BinaryVecScalarLessEqual:
  case BinaryVecScalarGreater:
  case BinaryVecScalarGreaterEqual:
  case BinaryVecScalarMin:
  case BinaryVecScalarMax:
  case UnaryNeg:
  case UnaryAbs:
  case UnarySqrt:
  case UnarySquare:
  case UnaryReciprocal:
  case UnarySign:
  case UnaryExp:
  case UnaryLog:
  case UnaryLog2:
  case UnaryLog10:
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
  // Extended binary vec-vec operators (bitwise, logical, special math)
  case BinaryVecVecBitwiseAnd:
  case BinaryVecVecBitwiseOr:
  case BinaryVecVecBitwiseXor:
  case BinaryVecVecLeftShift:
  case BinaryVecVecRightShift:
  case BinaryVecVecLogicalAnd:
  case BinaryVecVecLogicalOr:
  case BinaryVecVecLogicalXor:
  case BinaryVecVecAtan2:
  case BinaryVecVecHypot:
  case BinaryVecVecCopysign:
  case BinaryVecVecFmod:
  // Extended binary vec-scalar operators (bitwise, logical, special math)
  case BinaryVecScalarBitwiseAnd:
  case BinaryVecScalarBitwiseOr:
  case BinaryVecScalarBitwiseXor:
  case BinaryVecScalarLeftShift:
  case BinaryVecScalarRightShift:
  case BinaryVecScalarLogicalAnd:
  case BinaryVecScalarLogicalOr:
  case BinaryVecScalarLogicalXor:
  case BinaryVecScalarAtan2:
  case BinaryVecScalarHypot:
  case BinaryVecScalarCopysign:
  case BinaryVecScalarFmod:
  case BinaryVecScalarLeakyRelu:
  // Extended unary operators
  case UnaryExpm1:
  case UnaryExp2:
  case UnaryLog1p:
  case UnaryCbrt:
  case UnaryDegrees:
  case UnaryRadians:
  case UnaryLogicalNot:
  case UnaryBitwiseNot:
  case UnaryRelu:
  case UnarySigmoid:
  case UnaryGelu:
  case UnarySilu:
  case UnarySoftplus:
  case UnaryIsNan:
  case UnaryIsInf:
  // Ternary operators
  case TernaryClamp:
  case TernarySelect:
  // Reduction operations
  case ReduceSum:
  case ReduceMean:
  case ReduceMin:
  case ReduceMax:
  case ReduceProd:
  case ReduceAny:
  case ReduceAll:
  // Matrix operations
  case MatMul:
  case Transpose:
  case Dot:
  // Tensor creation operations
  case Arange:
  case Linspace:
  case Zeros:
  case Ones:
  case Full:
  // Norm operations
  case Norm:
  // Extended unary activations (Phase 1)
  case UnaryRelu6:
  case UnaryElu:
  case UnarySelu:
  case UnaryCelu:
  case UnaryMish:
  case UnaryHardswish:
  case UnaryHardsigmoid:
  case UnaryHardtanh:
  case UnarySoftsign:
  case UnaryLogSigmoid:
  case UnaryTanhshrink:
  // Extended unary math (Phase 2)
  case UnaryRsqrt:
  case UnaryTrunc:
  case UnaryFrac:
  case UnaryAsinh:
  case UnaryAcosh:
  case UnaryAtanh:
  case UnaryIsFinite:
  // Extended binary vec-vec (Phase 3)
  case BinaryVecVecLogaddexp:
  case BinaryVecVecLogaddexp2:
  // Extended binary vec-scalar activations
  case BinaryVecScalarPrelu:
  case BinaryVecScalarHardshrink:
  case BinaryVecScalarSoftshrink:
  case BinaryVecScalarLogaddexp:
  case BinaryVecScalarLogaddexp2:
  // Argmax/Argmin reductions
  case ReduceArgmax:
  case ReduceArgmin:
  case ReduceDimArgmax:
  case ReduceDimArgmin:
  // Cumulative scan operations
  case CumSum:
  case CumProd:
  // Prefix scan operations
  case PrefixScanExclusiveSum:
  case PrefixScanInclusiveSum:
  // Sort operations
  case SortBitonic:
  case SortRadix:
  // Dim-wise reductions
  case NormDim:
  case ReduceDimSum:
  case ReduceDimMean:
  case ReduceDimMin:
  case ReduceDimMax:
  case ReduceDimProd:
  case ReduceDimAny:
  case ReduceDimAll:
    return true;
  // Operators without shader support yet
  default:
    return false;
  }
}

// Generate shapes for testing given dimension count
// For multi-dimensional shapes, innermost dimension must be multiple of 4
// to avoid buffer padding issues with the current kernel implementation
std::vector<std::vector<uint32_t>> generateShapes(size_t numDims) {
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
uint32_t totalElements(const std::vector<uint32_t> &shape) {
  uint32_t total = 1;
  for (auto dim : shape) {
    total *= dim;
  }
  return total;
}

// Format shape as string for test naming
std::string shapeToString(const std::vector<uint32_t> &shape) {
  std::string result;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      result += "x";
    result += std::to_string(shape[i]);
  }
  return result;
}

// ============================================================================
// Reference Implementations for Operators
// ============================================================================

// Binary vec-vec reference
template <typename T>
T binaryVecVecRef(OperatorEnum op, T a, T b) {
  switch (op) {
  case BinaryVecVecAdd:
    return a + b;
  case BinaryVecVecSub:
    return a - b;
  case BinaryVecVecMul:
    return a * b;
  case BinaryVecVecDiv:
    return a / b;
  case BinaryVecVecMod:
    if constexpr (std::is_floating_point_v<T>) {
      return std::fmod(a, b);
    } else {
      return a % b;
    }
  case BinaryVecVecPow:
    if constexpr (std::is_floating_point_v<T>) {
      return std::pow(a, b);
    } else {
      return static_cast<T>(
          std::pow(static_cast<double>(a), static_cast<double>(b)));
    }
  case BinaryVecVecFloorDiv:
    if constexpr (std::is_floating_point_v<T>) {
      return std::floor(a / b);
    } else {
      return a / b;
    }
  case BinaryVecVecEqual:
    return static_cast<T>(a == b ? 1 : 0);
  case BinaryVecVecNotEqual:
    return static_cast<T>(a != b ? 1 : 0);
  case BinaryVecVecLess:
    return static_cast<T>(a < b ? 1 : 0);
  case BinaryVecVecLessEqual:
    return static_cast<T>(a <= b ? 1 : 0);
  case BinaryVecVecGreater:
    return static_cast<T>(a > b ? 1 : 0);
  case BinaryVecVecGreaterEqual:
    return static_cast<T>(a >= b ? 1 : 0);
  case BinaryVecVecMin:
    return std::min(a, b);
  case BinaryVecVecMax:
    return std::max(a, b);
  // Bitwise operations - for floats, match GPU shader behavior:
  // intBitsToFloat(floatBitsToInt(a) OP floatBitsToInt(b))
  case BinaryVecVecBitwiseAnd:
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
  case BinaryVecVecBitwiseOr:
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
  case BinaryVecVecBitwiseXor:
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
  case BinaryVecVecLeftShift:
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
  case BinaryVecVecRightShift:
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
  case BinaryVecVecLogicalAnd:
    return static_cast<T>((a != T{0} && b != T{0}) ? 1 : 0);
  case BinaryVecVecLogicalOr:
    return static_cast<T>((a != T{0} || b != T{0}) ? 1 : 0);
  case BinaryVecVecLogicalXor:
    return static_cast<T>((a != T{0}) != (b != T{0}) ? 1 : 0);
  // Math operations
  case BinaryVecVecAtan2:
    if constexpr (std::is_floating_point_v<T>) {
      return std::atan2(a, b);
    } else {
      return static_cast<T>(
          std::atan2(static_cast<double>(a), static_cast<double>(b)));
    }
  case BinaryVecVecHypot:
    if constexpr (std::is_floating_point_v<T>) {
      return std::hypot(a, b);
    } else {
      return static_cast<T>(
          std::hypot(static_cast<double>(a), static_cast<double>(b)));
    }
  case BinaryVecVecCopysign:
    if constexpr (std::is_floating_point_v<T>) {
      return std::copysign(a, b);
    } else if constexpr (std::is_signed_v<T>) {
      return b >= T{0} ? std::abs(a) : -std::abs(a);
    } else {
      return a; // unsigned values are always non-negative
    }
  case BinaryVecVecFmod:
    if constexpr (std::is_floating_point_v<T>) {
      return std::fmod(a, b);
    } else {
      return a % b;
    }
  case BinaryVecVecLogaddexp:
    if constexpr (std::is_floating_point_v<T>) {
      return std::max(a, b) + std::log(T{1} + std::exp(-std::abs(a - b)));
    } else {
      double da = static_cast<double>(a), db = static_cast<double>(b);
      return static_cast<T>(std::max(da, db) +
                            std::log(1.0 + std::exp(-std::abs(da - db))));
    }
  case BinaryVecVecLogaddexp2:
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
  case BinaryVecScalarAdd:
    return a + scalar;
  case BinaryVecScalarSub:
    return a - scalar;
  case BinaryVecScalarMul:
    return a * scalar;
  case BinaryVecScalarDiv:
    return a / scalar;
  case BinaryVecScalarMod:
    if constexpr (std::is_floating_point_v<T>) {
      return std::fmod(a, scalar);
    } else {
      return a % scalar;
    }
  case BinaryVecScalarPow:
    if constexpr (std::is_floating_point_v<T>) {
      return std::pow(a, scalar);
    } else {
      return static_cast<T>(
          std::pow(static_cast<double>(a), static_cast<double>(scalar)));
    }
  case BinaryVecScalarFloorDiv:
    if constexpr (std::is_floating_point_v<T>) {
      return std::floor(a / scalar);
    } else {
      return a / scalar;
    }
  case BinaryVecScalarEqual:
    return static_cast<T>(a == scalar ? 1 : 0);
  case BinaryVecScalarNotEqual:
    return static_cast<T>(a != scalar ? 1 : 0);
  case BinaryVecScalarLess:
    return static_cast<T>(a < scalar ? 1 : 0);
  case BinaryVecScalarLessEqual:
    return static_cast<T>(a <= scalar ? 1 : 0);
  case BinaryVecScalarGreater:
    return static_cast<T>(a > scalar ? 1 : 0);
  case BinaryVecScalarGreaterEqual:
    return static_cast<T>(a >= scalar ? 1 : 0);
  case BinaryVecScalarMin:
    return std::min(a, scalar);
  case BinaryVecScalarMax:
    return std::max(a, scalar);
  // Bitwise operations — GPU shader uses int(scalar) (value truncation) for the
  // scalar, but floatBitsToInt(a) (bit reinterpretation) for the vector element
  case BinaryVecScalarBitwiseAnd:
    if constexpr (std::is_integral_v<T>) {
      return a & static_cast<T>(scalar);
    } else {
      int ia, ir;
      memcpy(&ia, &a, sizeof(int));
      ir = ia & static_cast<int>(scalar);
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  case BinaryVecScalarBitwiseOr:
    if constexpr (std::is_integral_v<T>) {
      return a | static_cast<T>(scalar);
    } else {
      int ia, ir;
      memcpy(&ia, &a, sizeof(int));
      ir = ia | static_cast<int>(scalar);
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  case BinaryVecScalarBitwiseXor:
    if constexpr (std::is_integral_v<T>) {
      return a ^ static_cast<T>(scalar);
    } else {
      int ia, ir;
      memcpy(&ia, &a, sizeof(int));
      ir = ia ^ static_cast<int>(scalar);
      T r;
      memcpy(&r, &ir, sizeof(T));
      return r;
    }
  case BinaryVecScalarLeftShift:
    if constexpr (std::is_integral_v<T>) {
      return a << static_cast<int>(scalar);
    } else {
      uint32_t ua, ur;
      memcpy(&ua, &a, sizeof(uint32_t));
      ur = ua << (static_cast<uint32_t>(static_cast<int>(scalar)) & 31u);
      T r;
      memcpy(&r, &ur, sizeof(T));
      return r;
    }
  case BinaryVecScalarRightShift:
    if constexpr (std::is_integral_v<T>) {
      return a >> static_cast<int>(scalar);
    } else {
      uint32_t ua, ur;
      memcpy(&ua, &a, sizeof(uint32_t));
      ur = ua >> (static_cast<uint32_t>(static_cast<int>(scalar)) & 31u);
      T r;
      memcpy(&r, &ur, sizeof(T));
      return r;
    }
  // Logical operations
  case BinaryVecScalarLogicalAnd:
    return static_cast<T>((a != T{0} && scalar != T{0}) ? 1 : 0);
  case BinaryVecScalarLogicalOr:
    return static_cast<T>((a != T{0} || scalar != T{0}) ? 1 : 0);
  case BinaryVecScalarLogicalXor:
    return static_cast<T>((a != T{0}) != (scalar != T{0}) ? 1 : 0);
  // Math operations
  case BinaryVecScalarAtan2:
    if constexpr (std::is_floating_point_v<T>) {
      return std::atan2(a, scalar);
    } else {
      return static_cast<T>(
          std::atan2(static_cast<double>(a), static_cast<double>(scalar)));
    }
  case BinaryVecScalarHypot:
    if constexpr (std::is_floating_point_v<T>) {
      return std::hypot(a, scalar);
    } else {
      return static_cast<T>(
          std::hypot(static_cast<double>(a), static_cast<double>(scalar)));
    }
  case BinaryVecScalarCopysign:
    if constexpr (std::is_floating_point_v<T>) {
      return std::copysign(a, scalar);
    } else if constexpr (std::is_unsigned_v<T>) {
      return a; // unsigned values are always non-negative
    } else {
      return scalar >= T{0} ? std::abs(a) : -std::abs(a);
    }
  case BinaryVecScalarFmod:
    if constexpr (std::is_floating_point_v<T>) {
      return std::fmod(a, scalar);
    } else {
      return a % static_cast<T>(scalar);
    }
  // Activation
  case BinaryVecScalarLeakyRelu:
    return a > T{0} ? a : a * scalar;
  // Parameterized activations
  case BinaryVecScalarPrelu:
    return a >= T{0} ? a : scalar * a;
  case BinaryVecScalarHardshrink:
    if constexpr (std::is_floating_point_v<T>) {
      return std::abs(a) > scalar ? a : T{0};
    } else {
      return std::abs(static_cast<double>(a)) > static_cast<double>(scalar)
                 ? a
                 : T{0};
    }
  case BinaryVecScalarSoftshrink:
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
  case BinaryVecScalarLogaddexp:
    if constexpr (std::is_floating_point_v<T>) {
      return std::max(a, scalar) +
             std::log(T{1} + std::exp(-std::abs(a - scalar)));
    } else {
      double da = static_cast<double>(a), ds = static_cast<double>(scalar);
      return static_cast<T>(std::max(da, ds) +
                            std::log(1.0 + std::exp(-std::abs(da - ds))));
    }
  case BinaryVecScalarLogaddexp2:
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

// ============================================================================
// Test Data Generation
// ============================================================================

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

// ============================================================================
// Parameterized Test Fixture
// ============================================================================

class RuntimeOperatorTest : public ::testing::Test {
protected:
  void SetUp() override { runtime_ = std::make_unique<Runtime>(); }

  void TearDown() override {
    runtime_->shutdown();
    runtime_.reset();
  }

  void initBackend(BackendType backend) {
    if (backend == BackendType::Vulkan) {
      if (!runtime_->isVulkanAvailable()) {
        GTEST_SKIP() << "Vulkan not available on this system";
      }
    }
    runtime_->init(backend);
  }

  std::unique_ptr<Runtime> runtime_;
};

// ============================================================================
// Vulkan Backend Tests - All Operators
// ============================================================================

class VulkanBackendTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

// Test binary vec-vec operators with Float32
TEST_F(VulkanBackendTest, BinaryVecVecOperators_Float32) {
  const DataType dtype = DataType::Float32;

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(float);

      auto dataA = generateTestData<float>(elements, 42);
      auto dataB = generateTestData<float>(elements, 123);

      for (OperatorEnum op : kBinaryVecVecOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
        auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
        auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

        runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                      ComputeBinding(1, bufferB),
                                      ComputeBinding(2, bufferOut)});

        std::vector<float> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          float expected = binaryVecVecRef(op, dataA[i], dataB[i]);
          // Bitwise ops on floats can produce NaN/Inf bit patterns
          if (std::isnan(expected) && std::isnan(output[i]))
            continue;
          if (std::isinf(expected) && std::isinf(output[i]) &&
              std::signbit(expected) == std::signbit(output[i]))
            continue;
          float tol = (op == BinaryVecVecPow)
                          ? std::max(1e-5f, std::abs(expected) * 1e-5f)
                          : 1e-5f;
          EXPECT_NEAR(output[i], expected, tol)
              << "Mismatch at index " << i << " for " << operatorName(op);
        }
      }
    }
  }
}

TEST_F(VulkanBackendTest, BinaryVecVecOperators_Int32) {
  const DataType dtype = DataType::Int32;

  // Ops that produce valid GLSL for ivec4
  constexpr std::array<OperatorEnum, 22> kInt32BinaryVecVecOps = {
      // Arithmetic
      BinaryVecVecAdd, BinaryVecVecSub, BinaryVecVecMul, BinaryVecVecDiv,
      BinaryVecVecMod, BinaryVecVecFloorDiv,
      // Comparison
      BinaryVecVecEqual, BinaryVecVecNotEqual, BinaryVecVecLess,
      BinaryVecVecLessEqual, BinaryVecVecGreater, BinaryVecVecGreaterEqual,
      // Min/Max
      BinaryVecVecMin, BinaryVecVecMax,
      // Bitwise
      BinaryVecVecBitwiseAnd, BinaryVecVecBitwiseOr, BinaryVecVecBitwiseXor,
      BinaryVecVecLeftShift, BinaryVecVecRightShift,
      // Logical
      BinaryVecVecLogicalAnd, BinaryVecVecLogicalOr, BinaryVecVecLogicalXor};

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(int32_t);

      auto dataA = generateTestData<int32_t>(elements, 42);
      auto dataB = generateTestData<int32_t>(elements, 123);

      // Clamp shift amounts to [0, 15] to avoid overflow
      std::vector<int32_t> dataBShift = dataB;
      for (auto &v : dataBShift) {
        v = v % 16;
      }

      for (OperatorEnum op : kInt32BinaryVecVecOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        const auto &rhsData =
            (op == BinaryVecVecLeftShift || op == BinaryVecVecRightShift)
                ? dataBShift
                : dataB;

        auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
        auto bufferB = runtime_->createTensor(shape, dtype, rhsData.data());
        auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

        runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                      ComputeBinding(1, bufferB),
                                      ComputeBinding(2, bufferOut)});

        std::vector<int32_t> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          int32_t expected = binaryVecVecRef(op, dataA[i], rhsData[i]);
          EXPECT_EQ(output[i], expected)
              << "Mismatch at index " << i << " for " << operatorName(op);
        }
      }
    }
  }
}

TEST_F(VulkanBackendTest, BinaryVecVecOperators_UInt32) {
  const DataType dtype = DataType::UInt32;

  // Ops that produce valid GLSL for uvec4
  constexpr std::array<OperatorEnum, 22> kUInt32BinaryVecVecOps = {
      // Arithmetic
      BinaryVecVecAdd, BinaryVecVecSub, BinaryVecVecMul, BinaryVecVecDiv,
      BinaryVecVecMod, BinaryVecVecFloorDiv,
      // Comparison
      BinaryVecVecEqual, BinaryVecVecNotEqual, BinaryVecVecLess,
      BinaryVecVecLessEqual, BinaryVecVecGreater, BinaryVecVecGreaterEqual,
      // Min/Max
      BinaryVecVecMin, BinaryVecVecMax,
      // Bitwise
      BinaryVecVecBitwiseAnd, BinaryVecVecBitwiseOr, BinaryVecVecBitwiseXor,
      BinaryVecVecLeftShift, BinaryVecVecRightShift,
      // Logical
      BinaryVecVecLogicalAnd, BinaryVecVecLogicalOr, BinaryVecVecLogicalXor};

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(uint32_t);

      auto dataA = generateTestData<uint32_t>(elements, 42);
      auto dataB = generateTestData<uint32_t>(elements, 123);

      // Clamp shift amounts to [0, 15] to avoid overflow
      std::vector<uint32_t> dataBShift = dataB;
      for (auto &v : dataBShift) {
        v = v % 16;
      }

      for (OperatorEnum op : kUInt32BinaryVecVecOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        const auto &rhsData =
            (op == BinaryVecVecLeftShift || op == BinaryVecVecRightShift)
                ? dataBShift
                : dataB;

        auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
        auto bufferB = runtime_->createTensor(shape, dtype, rhsData.data());
        auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

        runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                      ComputeBinding(1, bufferB),
                                      ComputeBinding(2, bufferOut)});

        std::vector<uint32_t> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          uint32_t expected = binaryVecVecRef(op, dataA[i], rhsData[i]);
          EXPECT_EQ(output[i], expected)
              << "Mismatch at index " << i << " for " << operatorName(op);
        }
      }
    }
  }
}

// Test unary operators with Float32
TEST_F(VulkanBackendTest, UnaryOperators_Float32) {
  const DataType dtype = DataType::Float32;

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(float);

      auto dataIn = generateTestData<float>(elements, 42);
      for (auto &v : dataIn) {
        v = std::clamp(v, 0.1f, 0.9f);
      }

      for (OperatorEnum op : kUnaryOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
        auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

        runtime_->encodeOperator(
            op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

        std::vector<float> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          float expected = unaryRef(op, dataIn[i]);
          if (std::isfinite(expected)) {
            EXPECT_NEAR(output[i], expected, 1e-4f)
                << "Mismatch at index " << i << " for " << operatorName(op);
          }
        }
      }
    }
  }
}

// Test unary operators with Int32
TEST_F(VulkanBackendTest, UnaryOperators_Int32) {
  const DataType dtype = DataType::Int32;

  // Unary ops that produce valid GLSL for ivec4
  constexpr std::array<OperatorEnum, 11> kInt32UnaryOps = {
      UnaryNeg,        UnaryAbs,        UnarySquare, UnaryReciprocal,
      UnarySign,       UnaryFloor,      UnaryCeil,   UnaryRound,
      UnaryLogicalNot, UnaryBitwiseNot, UnaryRelu};

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(int32_t);

      auto dataIn = generateTestData<int32_t>(elements, 42);

      for (OperatorEnum op : kInt32UnaryOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
        auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

        runtime_->encodeOperator(
            op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

        std::vector<int32_t> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          int32_t expected = unaryRef(op, dataIn[i]);
          EXPECT_EQ(output[i], expected)
              << "Mismatch at index " << i << " for " << operatorName(op);
        }
      }
    }
  }
}

// Test binary vec-scalar operators with Float32
TEST_F(VulkanBackendTest, BinaryVecScalarOperators_Float32) {
  const DataType dtype = DataType::Float32;
  const float scalar = 2.5f;

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(float);

      auto dataA = generateTestData<float>(elements, 42);

      for (OperatorEnum op : kBinaryVecScalarOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
        auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

        runtime_->encodeOperator(
            op, {ComputeBinding(0, bufferA), ComputeBinding(1, bufferOut),
                 ComputeBinding(2, DataReference(scalar))});

        std::vector<float> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          float expected = binaryVecScalarRef(op, dataA[i], scalar);
          if (std::isnan(expected) && std::isnan(output[i]))
            continue;
          if (std::isinf(expected) && std::isinf(output[i]) &&
              std::signbit(expected) == std::signbit(output[i]))
            continue;
          float tol = (op == BinaryVecScalarPow)
                          ? std::max(1e-5f, std::abs(expected) * 1e-5f)
                          : 1e-5f;
          EXPECT_NEAR(output[i], expected, tol)
              << "Mismatch at index " << i << " for " << operatorName(op);
        }
      }
    }
  }
}

// ============================================================================
// Cross-Backend Consistency Tests
// ============================================================================

class DimensionSizeRangeTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(DimensionSizeRangeTest, AllSizes_1D_Float32) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecAdd;

  for (uint32_t size = kMinDimSize; size <= kMaxDimSize; ++size) {
    std::vector<uint32_t> shape = {size};
    const uint32_t elements = size;
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Size: " + std::to_string(size));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferB),
                                  ComputeBinding(2, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] + dataB[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for size " << size;
    }
  }
}

// Test 2D shapes with various outer dimensions and aligned inner dimension
TEST_F(DimensionSizeRangeTest, AllSizes_2D_Float32) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecMul;

  // Outer dimension (h) can be any odd value, inner dimension (w) must be
  // multiple of 4
  constexpr std::array<uint32_t, 6> outerSizes = {1, 3, 7, 9, 13, 17};
  constexpr std::array<uint32_t, 4> innerSizes = {4, 8, 12, 16};
  for (uint32_t h : outerSizes) {
    for (uint32_t w : innerSizes) {
      std::vector<uint32_t> shape = {h, w};
      const uint32_t elements = h * w;
      const size_t bufferSize = elements * sizeof(float);

      SCOPED_TRACE("Shape: " + std::to_string(h) + "x" + std::to_string(w));

      auto dataA = generateTestData<float>(elements, 42);
      auto dataB = generateTestData<float>(elements, 123);

      auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
      auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
      auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

      runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                    ComputeBinding(1, bufferB),
                                    ComputeBinding(2, bufferOut)});

      std::vector<float> output(elements);
      runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

      for (uint32_t i = 0; i < elements; ++i) {
        float expected = dataA[i] * dataB[i];
        EXPECT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
      }
    }
  }
}

// Test 3D shapes with various outer dimensions and aligned inner dimension
TEST_F(DimensionSizeRangeTest, AllSizes_3D_Float32) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecSub;

  // Outer dimensions can be odd values, innermost must be multiple of 4
  constexpr std::array<uint32_t, 4> outerSizes = {1, 3, 7, 9};
  constexpr std::array<uint32_t, 4> innerSizes = {4, 8, 12, 16};
  for (uint32_t d : outerSizes) {
    for (uint32_t h : outerSizes) {
      for (uint32_t w : innerSizes) {
        std::vector<uint32_t> shape = {d, h, w};
        const uint32_t elements = d * h * w;
        const size_t bufferSize = elements * sizeof(float);

        SCOPED_TRACE("Shape: " + std::to_string(d) + "x" + std::to_string(h) +
                     "x" + std::to_string(w));

        auto dataA = generateTestData<float>(elements, 42);
        auto dataB = generateTestData<float>(elements, 123);

        auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
        auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
        auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

        runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                      ComputeBinding(1, bufferB),
                                      ComputeBinding(2, bufferOut)});

        std::vector<float> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          float expected = dataA[i] - dataB[i];
          EXPECT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
        }
      }
    }
  }
}

// Test 4D shapes with various outer dimensions and aligned inner dimension
TEST_F(DimensionSizeRangeTest, AllSizes_4D_Float32) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecDiv;

  // Outer dimensions can be odd values, innermost must be multiple of 4
  constexpr std::array<uint32_t, 3> outerSizes = {1, 3, 7};
  constexpr std::array<uint32_t, 4> innerSizes = {4, 8, 12, 16};
  for (uint32_t n : outerSizes) {
    for (uint32_t c : outerSizes) {
      for (uint32_t h : outerSizes) {
        for (uint32_t w : innerSizes) {
          std::vector<uint32_t> shape = {n, c, h, w};
          const uint32_t elements = n * c * h * w;
          const size_t bufferSize = elements * sizeof(float);

          SCOPED_TRACE("Shape: " + std::to_string(n) + "x" + std::to_string(c) +
                       "x" + std::to_string(h) + "x" + std::to_string(w));

          auto dataA = generateTestData<float>(elements, 42);
          auto dataB = generateTestData<float>(elements, 123);
          // Avoid division by zero
          for (auto &v : dataB) {
            if (v < 0.1f)
              v = 0.1f;
          }

          auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
          auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
          auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

          runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                        ComputeBinding(1, bufferB),
                                        ComputeBinding(2, bufferOut)});

          std::vector<float> output(elements);
          runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

          for (uint32_t i = 0; i < elements; ++i) {
            float expected = dataA[i] / dataB[i];
            EXPECT_NEAR(output[i], expected, 1e-4f)
                << "Mismatch at index " << i;
          }
        }
      }
    }
  }
}

// ============================================================================
// Non-Aligned Innermost Dimension Tests
// ============================================================================

// Tests for multi-dimensional shapes where innermost dimension is NOT a
// multiple of 4 (1, 3, 5, 11, 13). These tests verify that the
// calculateAlignedElements function properly handles alignment.

class NonAlignedInnermostTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

// Test 2D shapes with non-aligned innermost dimensions
TEST_F(NonAlignedInnermostTest, BinaryVecVec_2D_InnermostDim1) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecAdd;

  for (uint32_t outer : {2u, 5u, 7u}) {
    std::vector<uint32_t> shape = {outer, 1};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferB),
                                  ComputeBinding(2, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] + dataB[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(NonAlignedInnermostTest, BinaryVecVec_2D_InnermostDim3) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecMul;

  for (uint32_t outer : {2u, 5u, 7u}) {
    std::vector<uint32_t> shape = {outer, 3};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferB),
                                  ComputeBinding(2, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] * dataB[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(NonAlignedInnermostTest, BinaryVecVec_2D_InnermostDim5) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecSub;

  for (uint32_t outer : {2u, 5u, 7u}) {
    std::vector<uint32_t> shape = {outer, 5};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferB),
                                  ComputeBinding(2, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] - dataB[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(NonAlignedInnermostTest, BinaryVecVec_2D_InnermostDim11) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecAdd;

  for (uint32_t outer : {2u, 3u, 5u}) {
    std::vector<uint32_t> shape = {outer, 11};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferB),
                                  ComputeBinding(2, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] + dataB[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(NonAlignedInnermostTest, BinaryVecVec_2D_InnermostDim13) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecMul;

  for (uint32_t outer : {2u, 3u, 5u}) {
    std::vector<uint32_t> shape = {outer, 13};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferB),
                                  ComputeBinding(2, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] * dataB[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Test 3D shapes with non-aligned innermost dimensions
TEST_F(NonAlignedInnermostTest, BinaryVecVec_3D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecAdd;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {2, 3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferB),
                                  ComputeBinding(2, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] + dataB[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Test 4D shapes with non-aligned innermost dimensions
TEST_F(NonAlignedInnermostTest, BinaryVecVec_4D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecMul;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {2, 2, 3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferB),
                                  ComputeBinding(2, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] * dataB[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Test unary operators with non-aligned innermost dimensions
TEST_F(NonAlignedInnermostTest, Unary_2D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = UnaryNeg;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {4, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataIn = generateTestData<float>(elements, 42);

    auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(
        op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = -dataIn[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Test vec-scalar operators with non-aligned innermost dimensions
TEST_F(NonAlignedInnermostTest, VecScalar_2D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecScalarMul;
  const float scalar = 2.5f;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {4, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferOut),
                                  ComputeBinding(2, DataReference(scalar))});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] * scalar;
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Vulkan backend tests with non-aligned innermost dimensions
class VulkanNonAlignedInnermostTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(VulkanNonAlignedInnermostTest, BinaryVecVec_2D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecAdd;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    for (uint32_t outer : {2u, 3u, 5u}) {
      std::vector<uint32_t> shape = {outer, innerDim};
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(float);

      SCOPED_TRACE("Shape: " + shapeToString(shape));

      auto dataA = generateTestData<float>(elements, 42);
      auto dataB = generateTestData<float>(elements, 123);

      auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
      auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
      auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

      runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                    ComputeBinding(1, bufferB),
                                    ComputeBinding(2, bufferOut)});

      std::vector<float> output(elements);
      runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

      for (uint32_t i = 0; i < elements; ++i) {
        float expected = dataA[i] + dataB[i];
        EXPECT_NEAR(output[i], expected, 1e-5f)
            << "Mismatch at index " << i << " for shape "
            << shapeToString(shape);
      }
    }
  }
}

TEST_F(VulkanNonAlignedInnermostTest, BinaryVecVec_3D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecMul;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {2, 3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferB),
                                  ComputeBinding(2, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] * dataB[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(VulkanNonAlignedInnermostTest, BinaryVecVec_4D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryVecVecSub;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {2, 2, 3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufferA),
                                  ComputeBinding(1, bufferB),
                                  ComputeBinding(2, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] - dataB[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(VulkanNonAlignedInnermostTest, Unary_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = UnarySquare;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataIn = generateTestData<float>(elements, 42);

    auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
    auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

    runtime_->encodeOperator(
        op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataIn[i] * dataIn[i];
      EXPECT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Test reduction operators with Float32 on Vulkan
TEST_F(VulkanBackendTest, ReductionOperators_Float32) {
  const DataType dtype = DataType::Float32;

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(float);

      auto dataIn = generateTestData<float>(elements, 42);

      for (OperatorEnum op : kReductionOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
        auto bufferOut = runtime_->createTensorEmpty({1}, dtype);

        // Initialize output to identity element
        float initVal = 0.0f;
        if (op == ReduceProd)
          initVal = 1.0f;
        else if (op == ReduceMin)
          initVal = std::numeric_limits<float>::max();
        else if (op == ReduceMax)
          initVal = std::numeric_limits<float>::lowest();
        else if (op == ReduceAll)
          initVal = 1.0f;
        runtime_->copyToTensor(bufferOut, &initVal, sizeof(float));

        runtime_->encodeOperator(
            op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

        float output = 0.0f;
        runtime_->copyFromTensor(bufferOut, &output, sizeof(float));

        // Verify result
        float expected = reduceRef(op, dataIn);
        if (std::isinf(expected) && std::isinf(output) &&
            std::signbit(expected) == std::signbit(output)) {
          // Both are same-sign infinity — pass
        } else if (op == ReduceMean || op == ReduceSum || op == ReduceProd) {
          EXPECT_NEAR(output, expected, std::abs(expected) * 1e-4f + 1e-5f)
              << "Mismatch for " << operatorName(op);
        } else {
          EXPECT_NEAR(output, expected, 1e-5f)
              << "Mismatch for " << operatorName(op);
        }
      }
    }
  }
}

// Test ternary clamp operator with Float32 on Vulkan
TEST_F(VulkanBackendTest, TernaryClamp_Float32) {
  const DataType dtype = DataType::Float32;

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(float);

      auto dataIn = generateTestData<float>(elements, 42);
      float minVal = 2.0f;
      float maxVal = 8.0f;

      SCOPED_TRACE(std::string("Shape: ") + shapeToString(shape));

      auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
      auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

      // For TernaryClamp, min and max are passed as data binding
      float clampVals[2] = {minVal, maxVal};
      runtime_->encodeOperator(
          TernaryClamp,
          {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
           ComputeBinding(2, DataReference(clampVals, sizeof(clampVals)))});

      std::vector<float> output(elements);
      runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

      // Verify results
      for (uint32_t i = 0; i < elements; ++i) {
        float expected = ternaryClampRef(dataIn[i], minVal, maxVal);
        EXPECT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
      }
    }
  }
}

// Test ternary select operator with Float32 on Vulkan
TEST_F(VulkanBackendTest, TernarySelect_Float32) {
  const DataType dtype = DataType::Float32;

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(float);

      auto dataCond = generateTestData<float>(elements, 42);
      auto dataX = generateTestData<float>(elements, 123);
      auto dataY = generateTestData<float>(elements, 456);

      // Make condition more varied (some zeros, some non-zeros)
      for (size_t i = 0; i < dataCond.size(); ++i) {
        dataCond[i] = (i % 3 == 0) ? 0.0f : dataCond[i];
      }

      SCOPED_TRACE(std::string("Shape: ") + shapeToString(shape));

      auto bufferCond = runtime_->createTensor(shape, dtype, dataCond.data());
      auto bufferX = runtime_->createTensor(shape, dtype, dataX.data());
      auto bufferY = runtime_->createTensor(shape, dtype, dataY.data());
      auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

      runtime_->encodeOperator(TernarySelect, {ComputeBinding(0, bufferCond),
                                               ComputeBinding(1, bufferX),
                                               ComputeBinding(2, bufferY),
                                               ComputeBinding(3, bufferOut)});

      std::vector<float> output(elements);
      runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

      // Verify results
      for (uint32_t i = 0; i < elements; ++i) {
        float expected = ternarySelectRef(dataCond[i], dataX[i], dataY[i]);
        EXPECT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
      }
    }
  }
}

// ============================================================================
// Dimension-wise Reduction Tests
// ============================================================================

// Dim reduction operators
constexpr std::array<OperatorEnum, 7> kDimReductionOps = {
    ReduceDimSum,  ReduceDimMean, ReduceDimMin, ReduceDimMax,
    ReduceDimProd, ReduceDimAny,  ReduceDimAll};

// CPU reference for dimension-wise reduction.
// Reduces data of shape (outerSize, reduceSize, innerSize) along the middle
// dimension, producing outerSize * innerSize outputs.
std::vector<float> dimReduceRef(OperatorEnum op,
                                const std::vector<float> &data,
                                uint32_t outerSize,
                                uint32_t reduceSize,
                                uint32_t innerSize) {
  std::vector<float> output(outerSize * innerSize);
  for (uint32_t o = 0; o < outerSize; ++o) {
    for (uint32_t i = 0; i < innerSize; ++i) {
      float val;
      switch (op) {
      case ReduceDimMin:
        val = std::numeric_limits<float>::max();
        break;
      case ReduceDimMax:
        val = std::numeric_limits<float>::lowest();
        break;
      case ReduceDimProd:
      case ReduceDimAll:
        val = 1.0f;
        break;
      default:
        val = 0.0f;
        break;
      }
      for (uint32_t r = 0; r < reduceSize; ++r) {
        float elem = data[o * reduceSize * innerSize + r * innerSize + i];
        switch (op) {
        case ReduceDimSum:
        case ReduceDimMean:
          val += elem;
          break;
        case ReduceDimMin:
          val = std::min(val, elem);
          break;
        case ReduceDimMax:
          val = std::max(val, elem);
          break;
        case ReduceDimProd:
          val *= elem;
          break;
        case ReduceDimAny:
          val = (val != 0.0f || elem != 0.0f) ? 1.0f : 0.0f;
          break;
        case ReduceDimAll:
          val = (val != 0.0f && elem != 0.0f) ? 1.0f : 0.0f;
          break;
        default:
          break;
        }
      }
      if (op == ReduceDimMean) {
        val /= static_cast<float>(reduceSize);
      }
      output[o * innerSize + i] = val;
    }
  }
  return output;
}

// CPU reference for dimension-wise L2 norm.
std::vector<float> normDimRef(const std::vector<float> &data,
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

// Test dim reduction operators on 2D tensors reducing along dim 0
TEST_F(VulkanBackendTest, DimReductionOperators_2D_Dim0) {
  const DataType dtype = DataType::Float32;

  struct TestCase {
    uint32_t rows;
    uint32_t cols;
  };
  constexpr std::array<TestCase, 4> testCases = {
      {{3, 4}, {7, 8}, {4, 12}, {13, 16}}};

  for (const auto &tc : testCases) {
    const uint32_t elements = tc.rows * tc.cols;
    auto dataIn = generateTestData<float>(elements, 42);

    // Reduce along dim 0: outerSize=1, reduceSize=rows, innerSize=cols
    uint32_t outerSize = 1;
    uint32_t reduceSize = tc.rows;
    uint32_t innerSize = tc.cols;

    for (OperatorEnum op : kDimReductionOps) {
      SCOPED_TRACE(std::string("Op: ") + operatorName(op) + " Shape: [" +
                   std::to_string(tc.rows) + ", " + std::to_string(tc.cols) +
                   "] dim=0");

      auto bufferIn =
          runtime_->createTensor({tc.rows, tc.cols}, dtype, dataIn.data());
      auto bufferOut = runtime_->createTensorEmpty({tc.cols}, dtype);

      uint32_t shapeData[3] = {outerSize, reduceSize, innerSize};
      runtime_->encodeOperator(
          op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
               ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

      std::vector<float> output(innerSize);
      runtime_->copyFromTensor(bufferOut, output.data(),
                               innerSize * sizeof(float));

      auto expected =
          dimReduceRef(op, dataIn, outerSize, reduceSize, innerSize);
      for (uint32_t i = 0; i < innerSize; ++i) {
        EXPECT_NEAR(output[i], expected[i],
                    std::abs(expected[i]) * 1e-4f + 1e-5f)
            << "Mismatch at index " << i;
      }
    }
  }
}

// Test dim reduction operators on 2D tensors reducing along dim 1
TEST_F(VulkanBackendTest, DimReductionOperators_2D_Dim1) {
  const DataType dtype = DataType::Float32;

  struct TestCase {
    uint32_t rows;
    uint32_t cols;
  };
  constexpr std::array<TestCase, 4> testCases = {
      {{3, 4}, {7, 8}, {4, 12}, {13, 16}}};

  for (const auto &tc : testCases) {
    const uint32_t elements = tc.rows * tc.cols;
    auto dataIn = generateTestData<float>(elements, 42);

    // Reduce along dim 1: outerSize=rows, reduceSize=cols, innerSize=1
    uint32_t outerSize = tc.rows;
    uint32_t reduceSize = tc.cols;
    uint32_t innerSize = 1;

    for (OperatorEnum op : kDimReductionOps) {
      SCOPED_TRACE(std::string("Op: ") + operatorName(op) + " Shape: [" +
                   std::to_string(tc.rows) + ", " + std::to_string(tc.cols) +
                   "] dim=1");

      auto bufferIn =
          runtime_->createTensor({tc.rows, tc.cols}, dtype, dataIn.data());
      auto bufferOut = runtime_->createTensorEmpty({tc.rows}, dtype);

      uint32_t shapeData[3] = {outerSize, reduceSize, innerSize};
      runtime_->encodeOperator(
          op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
               ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

      std::vector<float> output(outerSize);
      runtime_->copyFromTensor(bufferOut, output.data(),
                               outerSize * sizeof(float));

      auto expected =
          dimReduceRef(op, dataIn, outerSize, reduceSize, innerSize);
      for (uint32_t i = 0; i < outerSize; ++i) {
        EXPECT_NEAR(output[i], expected[i],
                    std::abs(expected[i]) * 1e-4f + 1e-5f)
            << "Mismatch at index " << i;
      }
    }
  }
}

// Test dim reduction on 3D tensor reducing along the middle dimension
TEST_F(VulkanBackendTest, DimReductionOperators_3D_MiddleDim) {
  const DataType dtype = DataType::Float32;
  const uint32_t d0 = 3, d1 = 5, d2 = 4;
  const uint32_t elements = d0 * d1 * d2;
  auto dataIn = generateTestData<float>(elements, 42);

  // Reduce along dim 1: outerSize=d0, reduceSize=d1, innerSize=d2
  uint32_t outerSize = d0;
  uint32_t reduceSize = d1;
  uint32_t innerSize = d2;
  uint32_t numOutputs = outerSize * innerSize;

  for (OperatorEnum op : kDimReductionOps) {
    SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                 " Shape: [3, 5, 4] dim=1");

    auto bufferIn = runtime_->createTensor({d0, d1, d2}, dtype, dataIn.data());
    auto bufferOut = runtime_->createTensorEmpty({d0, d2}, dtype);

    uint32_t shapeData[3] = {outerSize, reduceSize, innerSize};
    runtime_->encodeOperator(
        op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
             ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

    std::vector<float> output(numOutputs);
    runtime_->copyFromTensor(bufferOut, output.data(),
                             numOutputs * sizeof(float));

    auto expected = dimReduceRef(op, dataIn, outerSize, reduceSize, innerSize);
    for (uint32_t i = 0; i < numOutputs; ++i) {
      EXPECT_NEAR(output[i], expected[i], std::abs(expected[i]) * 1e-4f + 1e-5f)
          << "Mismatch at index " << i;
    }
  }
}

// ============================================================================
// Dimension-wise Norm Tests
// ============================================================================

// Test NormDim on 2D tensors reducing along dim 0
TEST_F(VulkanBackendTest, NormDim_2D_Dim0) {
  const DataType dtype = DataType::Float32;

  struct TestCase {
    uint32_t rows;
    uint32_t cols;
  };
  constexpr std::array<TestCase, 4> testCases = {
      {{3, 4}, {7, 8}, {4, 12}, {13, 16}}};

  for (const auto &tc : testCases) {
    const uint32_t elements = tc.rows * tc.cols;
    auto dataIn = generateTestData<float>(elements, 42);

    uint32_t outerSize = 1;
    uint32_t reduceSize = tc.rows;
    uint32_t innerSize = tc.cols;

    SCOPED_TRACE(std::string("Shape: [") + std::to_string(tc.rows) + ", " +
                 std::to_string(tc.cols) + "] dim=0");

    auto bufferIn =
        runtime_->createTensor({tc.rows, tc.cols}, dtype, dataIn.data());
    auto bufferOut = runtime_->createTensorEmpty({tc.cols}, dtype);

    uint32_t shapeData[3] = {outerSize, reduceSize, innerSize};
    runtime_->encodeOperator(
        NormDim,
        {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
         ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

    std::vector<float> output(innerSize);
    runtime_->copyFromTensor(bufferOut, output.data(),
                             innerSize * sizeof(float));

    auto expected = normDimRef(dataIn, outerSize, reduceSize, innerSize);
    for (uint32_t i = 0; i < innerSize; ++i) {
      EXPECT_NEAR(output[i], expected[i], std::abs(expected[i]) * 1e-4f + 1e-5f)
          << "Mismatch at index " << i;
    }
  }
}

// Test NormDim on 2D tensors reducing along dim 1
TEST_F(VulkanBackendTest, NormDim_2D_Dim1) {
  const DataType dtype = DataType::Float32;

  struct TestCase {
    uint32_t rows;
    uint32_t cols;
  };
  constexpr std::array<TestCase, 4> testCases = {
      {{3, 4}, {7, 8}, {4, 12}, {13, 16}}};

  for (const auto &tc : testCases) {
    const uint32_t elements = tc.rows * tc.cols;
    auto dataIn = generateTestData<float>(elements, 42);

    uint32_t outerSize = tc.rows;
    uint32_t reduceSize = tc.cols;
    uint32_t innerSize = 1;

    SCOPED_TRACE(std::string("Shape: [") + std::to_string(tc.rows) + ", " +
                 std::to_string(tc.cols) + "] dim=1");

    auto bufferIn =
        runtime_->createTensor({tc.rows, tc.cols}, dtype, dataIn.data());
    auto bufferOut = runtime_->createTensorEmpty({tc.rows}, dtype);

    uint32_t shapeData[3] = {outerSize, reduceSize, innerSize};
    runtime_->encodeOperator(
        NormDim,
        {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
         ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

    std::vector<float> output(outerSize);
    runtime_->copyFromTensor(bufferOut, output.data(),
                             outerSize * sizeof(float));

    auto expected = normDimRef(dataIn, outerSize, reduceSize, innerSize);
    for (uint32_t i = 0; i < outerSize; ++i) {
      EXPECT_NEAR(output[i], expected[i], std::abs(expected[i]) * 1e-4f + 1e-5f)
          << "Mismatch at index " << i;
    }
  }
}

// Test NormDim on 3D tensor reducing along middle dimension
TEST_F(VulkanBackendTest, NormDim_3D_MiddleDim) {
  const DataType dtype = DataType::Float32;
  const uint32_t d0 = 3, d1 = 5, d2 = 4;
  const uint32_t elements = d0 * d1 * d2;
  auto dataIn = generateTestData<float>(elements, 42);

  uint32_t outerSize = d0;
  uint32_t reduceSize = d1;
  uint32_t innerSize = d2;
  uint32_t numOutputs = outerSize * innerSize;

  SCOPED_TRACE("Shape: [3, 5, 4] dim=1");

  auto bufferIn = runtime_->createTensor({d0, d1, d2}, dtype, dataIn.data());
  auto bufferOut = runtime_->createTensorEmpty({d0, d2}, dtype);

  uint32_t shapeData[3] = {outerSize, reduceSize, innerSize};
  runtime_->encodeOperator(
      NormDim,
      {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
       ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

  std::vector<float> output(numOutputs);
  runtime_->copyFromTensor(bufferOut, output.data(),
                           numOutputs * sizeof(float));

  auto expected = normDimRef(dataIn, outerSize, reduceSize, innerSize);
  for (uint32_t i = 0; i < numOutputs; ++i) {
    EXPECT_NEAR(output[i], expected[i], std::abs(expected[i]) * 1e-4f + 1e-5f)
        << "Mismatch at index " << i;
  }
}

// Test NormDim with known values (3-4-5 triangle)
TEST_F(VulkanBackendTest, NormDim_KnownValues) {
  const DataType dtype = DataType::Float32;

  // [[3, 5], [4, 12]] -> norm along dim 0 -> [sqrt(9+16), sqrt(25+144)]
  //                                        = [5, 13]
  std::vector<float> dataIn = {3.0f, 5.0f, 4.0f, 12.0f};
  uint32_t outerSize = 1;
  uint32_t reduceSize = 2;
  uint32_t innerSize = 2;

  auto bufferIn = runtime_->createTensor({2, 2}, dtype, dataIn.data());
  auto bufferOut = runtime_->createTensorEmpty({2}, dtype);

  uint32_t shapeData[3] = {outerSize, reduceSize, innerSize};
  runtime_->encodeOperator(
      NormDim,
      {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
       ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

  std::vector<float> output(2);
  runtime_->copyFromTensor(bufferOut, output.data(), 2 * sizeof(float));

  EXPECT_NEAR(output[0], 5.0f, 1e-5f);
  EXPECT_NEAR(output[1], 13.0f, 1e-5f);
}

// ============================================================================
// Runtime Lifecycle Tests
// ============================================================================

class RuntimeLifecycleTest : public ::testing::Test {};

TEST_F(RuntimeLifecycleTest, DefaultConstruction) {
  Runtime runtime;
  EXPECT_EQ(runtime.currentBackend(), BackendType::Vulkan);
}

TEST_F(RuntimeLifecycleTest, VulkanInitialization) {
  Runtime runtime;
  if (runtime.isVulkanAvailable()) {
    EXPECT_NO_THROW(runtime.init(BackendType::Vulkan));
    EXPECT_EQ(runtime.currentBackend(), BackendType::Vulkan);
    runtime.shutdown();
  } else {
    EXPECT_THROW(runtime.init(BackendType::Vulkan), std::runtime_error);
  }
}

TEST_F(RuntimeLifecycleTest, MultipleInitShutdown) {
  Runtime runtime;

  if (!runtime.isVulkanAvailable()) {
    GTEST_SKIP() << "Vulkan not available";
  }

  // First cycle
  runtime.init(BackendType::Vulkan);
  {
    auto buf1 = runtime.createTensorEmpty({16}, DataType::Float32);
    EXPECT_TRUE(buf1);
  } // buf1 goes out of scope here
  runtime.shutdown();

  // Second cycle
  runtime.init(BackendType::Vulkan);
  {
    auto buf2 = runtime.createTensorEmpty({16}, DataType::Float32);
    EXPECT_TRUE(buf2);
  } // buf2 goes out of scope here
  runtime.shutdown();
}

TEST_F(RuntimeLifecycleTest, MoveSemantics) {
  Runtime runtime1;

  if (!runtime1.isVulkanAvailable()) {
    GTEST_SKIP() << "Vulkan not available";
  }

  runtime1.init(BackendType::Vulkan);
  {
    auto buf = runtime1.createTensorEmpty({16}, DataType::Float32);
    EXPECT_TRUE(buf);
  } // buf goes out of scope here

  // Move construct
  Runtime runtime2 = std::move(runtime1);
  EXPECT_EQ(runtime2.currentBackend(), BackendType::Vulkan);

  // Move assign
  Runtime runtime3;
  runtime3 = std::move(runtime2);
  EXPECT_EQ(runtime3.currentBackend(), BackendType::Vulkan);

  runtime3.shutdown();
}

// ============================================================================
// Argmax/Argmin Tests
// ============================================================================

class ArgmaxArgminTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(ArgmaxArgminTest, GlobalArgmax_Float32) {
  const DataType dtype = DataType::Float32;
  std::vector<float> data = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 7.0f, 4.0f, 6.0f};
  const uint32_t elements = static_cast<uint32_t>(data.size());

  auto bufferIn = runtime_->createTensor({elements}, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty({1}, dtype);

  runtime_->encodeOperator(ReduceArgmax, {ComputeBinding(0, bufferIn),
                                          ComputeBinding(1, bufferOut)});

  float output = 0.0f;
  runtime_->copyFromTensor(bufferOut, &output, sizeof(float));

  EXPECT_EQ(static_cast<int>(output), 3)
      << "Argmax should be index 3 (value 9.0)";
}

TEST_F(ArgmaxArgminTest, GlobalArgmin_Float32) {
  const DataType dtype = DataType::Float32;
  std::vector<float> data = {5.0f, 3.0f, 1.0f, 9.0f, 2.0f, 7.0f, 4.0f, 6.0f};
  const uint32_t elements = static_cast<uint32_t>(data.size());

  auto bufferIn = runtime_->createTensor({elements}, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty({1}, dtype);

  runtime_->encodeOperator(ReduceArgmin, {ComputeBinding(0, bufferIn),
                                          ComputeBinding(1, bufferOut)});

  float output = 0.0f;
  runtime_->copyFromTensor(bufferOut, &output, sizeof(float));

  EXPECT_EQ(static_cast<int>(output), 2)
      << "Argmin should be index 2 (value 1.0)";
}

TEST_F(ArgmaxArgminTest, GlobalArgmax_LargeTensor) {
  const DataType dtype = DataType::Float32;
  const uint32_t elements = 1024;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

  std::vector<float> data(elements);
  for (auto &v : data)
    v = dist(gen);

  // Find expected argmax
  int expectedIdx = 0;
  for (uint32_t i = 1; i < elements; ++i) {
    if (data[i] > data[expectedIdx])
      expectedIdx = static_cast<int>(i);
  }

  auto bufferIn = runtime_->createTensor({elements}, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty({1}, dtype);

  runtime_->encodeOperator(ReduceArgmax, {ComputeBinding(0, bufferIn),
                                          ComputeBinding(1, bufferOut)});

  float output = 0.0f;
  runtime_->copyFromTensor(bufferOut, &output, sizeof(float));

  EXPECT_EQ(static_cast<int>(output), expectedIdx);
}

TEST_F(ArgmaxArgminTest, DimArgmax_2D_Dim0) {
  const DataType dtype = DataType::Float32;
  // 3x4 matrix, find argmax along dim 0 (across rows)
  std::vector<float> data = {
      1.0f, 9.0f, 3.0f, 2.0f, // row 0
      7.0f, 4.0f, 8.0f, 5.0f, // row 1
      6.0f, 2.0f, 1.0f, 10.0f // row 2
  };
  std::vector<uint32_t> shape = {3, 4};

  auto bufferIn = runtime_->createTensor(shape, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty({4}, dtype);

  // For dim 0: outerSize=1, reduceSize=3, innerSize=4
  uint32_t shapeData[5] = {
      1, 3, 4, 4, 4}; // outer, reduce, inner, inOuterStride, inReduceStride

  runtime_->encodeOperator(
      ReduceDimArgmax,
      {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
       ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

  std::vector<float> output(4);
  runtime_->copyFromTensor(bufferOut, output.data(), 4 * sizeof(float));

  // Expected: col 0->row 1 (7), col 1->row 0 (9), col 2->row 1 (8), col 3->row
  // 2 (10)
  EXPECT_EQ(static_cast<int>(output[0]), 1);
  EXPECT_EQ(static_cast<int>(output[1]), 0);
  EXPECT_EQ(static_cast<int>(output[2]), 1);
  EXPECT_EQ(static_cast<int>(output[3]), 2);
}

TEST_F(ArgmaxArgminTest, DimArgmin_2D_Dim0) {
  const DataType dtype = DataType::Float32;
  // Same data as above
  std::vector<float> data = {1.0f, 9.0f, 3.0f, 2.0f, 7.0f, 4.0f,
                             8.0f, 5.0f, 6.0f, 2.0f, 1.0f, 10.0f};
  std::vector<uint32_t> shape = {3, 4};

  auto bufferIn = runtime_->createTensor(shape, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty({4}, dtype);

  uint32_t shapeData[5] = {1, 3, 4, 4, 4};

  runtime_->encodeOperator(
      ReduceDimArgmin,
      {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
       ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

  std::vector<float> output(4);
  runtime_->copyFromTensor(bufferOut, output.data(), 4 * sizeof(float));

  // Expected: col 0->row 0 (1), col 1->row 2 (2), col 2->row 2 (1), col 3->row
  // 0 (2)
  EXPECT_EQ(static_cast<int>(output[0]), 0);
  EXPECT_EQ(static_cast<int>(output[1]), 2);
  EXPECT_EQ(static_cast<int>(output[2]), 2);
  EXPECT_EQ(static_cast<int>(output[3]), 0);
}

// ============================================================================
// Cumulative Scan Tests
// ============================================================================

class CumsumCumprodTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(CumsumCumprodTest, CumSum_1D) {
  const DataType dtype = DataType::Float32;
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const uint32_t elements = static_cast<uint32_t>(data.size());

  auto bufferIn = runtime_->createTensor({elements}, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

  // 1D cumsum: outerSize=1, reduceSize=8, innerSize=1
  uint32_t shapeData[5] = {1, 8, 1, 1, 1};

  runtime_->encodeOperator(
      CumSum, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
               ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  std::vector<float> expected = {1.0f,  3.0f,  6.0f,  10.0f,
                                 15.0f, 21.0f, 28.0f, 36.0f};
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-5f)
        << "CumSum mismatch at index " << i;
  }
}

TEST_F(CumsumCumprodTest, CumProd_1D) {
  const DataType dtype = DataType::Float32;
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const uint32_t elements = static_cast<uint32_t>(data.size());

  auto bufferIn = runtime_->createTensor({elements}, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

  uint32_t shapeData[5] = {1, 4, 1, 1, 1};

  runtime_->encodeOperator(
      CumProd,
      {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
       ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  std::vector<float> expected = {1.0f, 2.0f, 6.0f, 24.0f};
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-5f)
        << "CumProd mismatch at index " << i;
  }
}

TEST_F(CumsumCumprodTest, CumSum_2D_Dim0) {
  const DataType dtype = DataType::Float32;
  // 3x4 matrix, cumsum along dim 0
  std::vector<float> data = {
      1.0f, 2.0f,  3.0f,  4.0f, // row 0
      5.0f, 6.0f,  7.0f,  8.0f, // row 1
      9.0f, 10.0f, 11.0f, 12.0f // row 2
  };
  std::vector<uint32_t> shape = {3, 4};
  const uint32_t elements = 12;

  auto bufferIn = runtime_->createTensor(shape, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

  // dim 0: outerSize=1, reduceSize=3, innerSize=4
  uint32_t shapeData[5] = {1, 3, 4, 4, 4};

  runtime_->encodeOperator(
      CumSum, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
               ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  // Expected: cumsum along rows
  std::vector<float> expected = {
      1.0f,  2.0f,  3.0f,  4.0f,  // row 0
      6.0f,  8.0f,  10.0f, 12.0f, // row 0 + row 1
      15.0f, 18.0f, 21.0f, 24.0f  // row 0 + row 1 + row 2
  };
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-5f)
        << "CumSum 2D dim0 mismatch at index " << i;
  }
}

TEST_F(CumsumCumprodTest, CumProd_2D_Dim0) {
  const DataType dtype = DataType::Float32;
  // 2x4 matrix, cumprod along dim 0
  std::vector<float> data = {
      1.0f, 2.0f, 3.0f, 4.0f, // row 0
      5.0f, 6.0f, 7.0f, 8.0f  // row 1
  };
  std::vector<uint32_t> shape = {2, 4};
  const uint32_t elements = 8;

  auto bufferIn = runtime_->createTensor(shape, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

  // dim 0: outerSize=1, reduceSize=2, innerSize=4
  uint32_t shapeData[5] = {1, 2, 4, 4, 4};

  runtime_->encodeOperator(
      CumProd,
      {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
       ComputeBinding(2, DataReference(shapeData, sizeof(shapeData)))});

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  std::vector<float> expected = {
      1.0f, 2.0f,  3.0f,  4.0f, // row 0
      5.0f, 12.0f, 21.0f, 32.0f // row 0 * row 1
  };
  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-5f)
        << "CumProd 2D dim0 mismatch at index " << i;
  }
}

// ============================================================================
// Extended Activation & Math Shader Compilation Tests
// ============================================================================

class NewOpsShaderCompileTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(NewOpsShaderCompileTest, AllNewUnaryActivations_Compile) {
  const DataType dtype = DataType::Float32;
  constexpr std::array<OperatorEnum, 11> kNewUnaryActivations = {
      UnaryRelu6,    UnaryElu,        UnarySelu,        UnaryCelu,
      UnaryMish,     UnaryHardswish,  UnaryHardsigmoid, UnaryHardtanh,
      UnarySoftsign, UnaryLogSigmoid, UnaryTanhshrink};

  std::vector<float> dataIn = {-2.0f, -1.0f, 0.0f, 1.0f,
                               2.0f,  3.0f,  5.0f, 7.0f};
  const uint32_t elements = static_cast<uint32_t>(dataIn.size());
  const size_t bufferSize = elements * sizeof(float);

  for (OperatorEnum op : kNewUnaryActivations) {
    SCOPED_TRACE(std::string("Op: ") + operatorName(op));

    auto bufferIn = runtime_->createTensor({elements}, dtype, dataIn.data());
    auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

    EXPECT_NO_THROW(runtime_->encodeOperator(
        op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)}));

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = unaryRef(op, dataIn[i]);
      if (std::isfinite(expected)) {
        EXPECT_NEAR(output[i], expected, 1e-3f)
            << "Mismatch at index " << i << " for " << operatorName(op);
      }
    }
  }
}

TEST_F(NewOpsShaderCompileTest, AllNewUnaryMath_Compile) {
  const DataType dtype = DataType::Float32;
  constexpr std::array<OperatorEnum, 7> kNewUnaryMath = {
      UnaryRsqrt, UnaryTrunc, UnaryFrac,    UnaryAsinh,
      UnaryAcosh, UnaryAtanh, UnaryIsFinite};

  // Values suitable for all these functions (acosh requires >= 1, atanh
  // requires |x| < 1)
  std::vector<float> dataIn = {0.1f, 0.5f, 0.9f, 1.5f, 2.0f, 3.0f, 5.0f, 10.0f};
  const uint32_t elements = static_cast<uint32_t>(dataIn.size());
  const size_t bufferSize = elements * sizeof(float);

  for (OperatorEnum op : kNewUnaryMath) {
    SCOPED_TRACE(std::string("Op: ") + operatorName(op));

    auto bufferIn = runtime_->createTensor({elements}, dtype, dataIn.data());
    auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

    EXPECT_NO_THROW(runtime_->encodeOperator(
        op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)}));

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = unaryRef(op, dataIn[i]);
      if (std::isfinite(expected)) {
        EXPECT_NEAR(output[i], expected, 1e-4f)
            << "Mismatch at index " << i << " for " << operatorName(op);
      }
    }
  }
}

TEST_F(NewOpsShaderCompileTest, NewBinaryVecVec_Logaddexp) {
  const DataType dtype = DataType::Float32;
  std::vector<float> dataA = {1.0f,  2.0f, 3.0f, -1.0f,
                              -2.0f, 0.0f, 5.0f, 10.0f};
  std::vector<float> dataB = {2.0f, 1.0f, 3.0f, 0.0f, -3.0f, 0.0f, 4.0f, 9.0f};
  const uint32_t elements = static_cast<uint32_t>(dataA.size());
  const size_t bufferSize = elements * sizeof(float);

  for (OperatorEnum op : {BinaryVecVecLogaddexp, BinaryVecVecLogaddexp2}) {
    SCOPED_TRACE(std::string("Op: ") + operatorName(op));

    auto bufA = runtime_->createTensor({elements}, dtype, dataA.data());
    auto bufB = runtime_->createTensor({elements}, dtype, dataB.data());
    auto bufOut = runtime_->createTensorEmpty({elements}, dtype);

    runtime_->encodeOperator(op,
                             {ComputeBinding(0, bufA), ComputeBinding(1, bufB),
                              ComputeBinding(2, bufOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = binaryVecVecRef(op, dataA[i], dataB[i]);
      EXPECT_NEAR(output[i], expected, 1e-4f)
          << "Mismatch at index " << i << " for " << operatorName(op);
    }
  }
}

TEST_F(NewOpsShaderCompileTest, NewBinaryVecScalar_ParameterizedActivations) {
  const DataType dtype = DataType::Float32;
  std::vector<float> dataA = {-3.0f, -1.0f, -0.5f, 0.0f,
                              0.5f,  1.0f,  2.0f,  3.0f};
  const uint32_t elements = static_cast<uint32_t>(dataA.size());
  const size_t bufferSize = elements * sizeof(float);
  const float scalar = 0.5f;

  for (OperatorEnum op : {BinaryVecScalarPrelu, BinaryVecScalarHardshrink,
                          BinaryVecScalarSoftshrink, BinaryVecScalarLogaddexp,
                          BinaryVecScalarLogaddexp2}) {
    SCOPED_TRACE(std::string("Op: ") + operatorName(op));

    auto bufA = runtime_->createTensor({elements}, dtype, dataA.data());
    auto bufOut = runtime_->createTensorEmpty({elements}, dtype);

    runtime_->encodeOperator(op, {ComputeBinding(0, bufA),
                                  ComputeBinding(1, bufOut),
                                  ComputeBinding(2, DataReference(scalar))});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = binaryVecScalarRef(op, dataA[i], scalar);
      if (std::isfinite(expected)) {
        EXPECT_NEAR(output[i], expected, 1e-4f)
            << "Mismatch at index " << i << " for " << operatorName(op);
      }
    }
  }
}

// ============================================================================
// Multi-Workgroup Reduce Tests
// ============================================================================

class MultiWorkgroupReduceTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(MultiWorkgroupReduceTest, ReduceSum_LargeArray) {
  // Tests that trigger multi-workgroup reduce (>65536 elements)
  for (uint32_t elements : {257u, 1000u, 4096u, 65537u, 100000u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    auto bufIn =
        runtime_->createTensor({elements}, DataType::Float32, data.data());
    auto bufOut = runtime_->createTensorEmpty({1}, DataType::Float32);

    float initVal = 0.0f;
    runtime_->copyToTensor(bufOut, &initVal, sizeof(float));

    runtime_->encodeOperator(
        ReduceSum, {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut)});

    float output = 0.0f;
    runtime_->copyFromTensor(bufOut, &output, sizeof(float));

    float expected = reduceRef<float>(ReduceSum, data);
    EXPECT_NEAR(output, expected, std::abs(expected) * 1e-3f + 1e-3f)
        << "ReduceSum mismatch for " << elements << " elements";
  }
}

TEST_F(MultiWorkgroupReduceTest, ReduceMean_LargeArray) {
  for (uint32_t elements : {1000u, 65537u, 100000u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    auto bufIn =
        runtime_->createTensor({elements}, DataType::Float32, data.data());
    auto bufOut = runtime_->createTensorEmpty({1}, DataType::Float32);

    float initVal = 0.0f;
    runtime_->copyToTensor(bufOut, &initVal, sizeof(float));

    runtime_->encodeOperator(
        ReduceMean, {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut)});

    float output = 0.0f;
    runtime_->copyFromTensor(bufOut, &output, sizeof(float));

    float expected = reduceRef<float>(ReduceMean, data);
    EXPECT_NEAR(output, expected, std::abs(expected) * 1e-3f + 1e-3f)
        << "ReduceMean mismatch for " << elements << " elements";
  }
}

TEST_F(MultiWorkgroupReduceTest, ReduceMinMax_LargeArray) {
  uint32_t elements = 100000u;
  auto data = generateTestData<float>(elements, 42);
  auto bufIn =
      runtime_->createTensor({elements}, DataType::Float32, data.data());

  // Test ReduceMin
  {
    auto bufOut = runtime_->createTensorEmpty({1}, DataType::Float32);
    float initVal = std::numeric_limits<float>::max();
    runtime_->copyToTensor(bufOut, &initVal, sizeof(float));

    runtime_->encodeOperator(
        ReduceMin, {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut)});

    float output = 0.0f;
    runtime_->copyFromTensor(bufOut, &output, sizeof(float));

    float expected = reduceRef<float>(ReduceMin, data);
    EXPECT_NEAR(output, expected, 1e-5f) << "ReduceMin mismatch";
  }

  // Test ReduceMax
  {
    auto bufOut = runtime_->createTensorEmpty({1}, DataType::Float32);
    float initVal = std::numeric_limits<float>::lowest();
    runtime_->copyToTensor(bufOut, &initVal, sizeof(float));

    runtime_->encodeOperator(
        ReduceMax, {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut)});

    float output = 0.0f;
    runtime_->copyFromTensor(bufOut, &output, sizeof(float));

    float expected = reduceRef<float>(ReduceMax, data);
    EXPECT_NEAR(output, expected, 1e-5f) << "ReduceMax mismatch";
  }
}

TEST_F(MultiWorkgroupReduceTest, ReduceProd_LargeArray) {
  // Use smaller values to avoid overflow
  uint32_t elements = 1000u;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.99f, 1.01f);
  std::vector<float> data(elements);
  for (auto &v : data)
    v = dist(gen);

  auto bufIn =
      runtime_->createTensor({elements}, DataType::Float32, data.data());
  auto bufOut = runtime_->createTensorEmpty({1}, DataType::Float32);

  float initVal = 1.0f;
  runtime_->copyToTensor(bufOut, &initVal, sizeof(float));

  runtime_->encodeOperator(
      ReduceProd, {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut)});

  float output = 0.0f;
  runtime_->copyFromTensor(bufOut, &output, sizeof(float));

  float expected = reduceRef<float>(ReduceProd, data);
  EXPECT_NEAR(output, expected, std::abs(expected) * 1e-2f + 1e-5f)
      << "ReduceProd mismatch";
}

TEST_F(MultiWorkgroupReduceTest, SmallArrayStillWorks) {
  // Verify small arrays (<=256) still use single-workgroup path
  for (uint32_t elements : {1u, 4u, 100u, 256u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    auto bufIn =
        runtime_->createTensor({elements}, DataType::Float32, data.data());
    auto bufOut = runtime_->createTensorEmpty({1}, DataType::Float32);

    float initVal = 0.0f;
    runtime_->copyToTensor(bufOut, &initVal, sizeof(float));

    runtime_->encodeOperator(
        ReduceSum, {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut)});

    float output = 0.0f;
    runtime_->copyFromTensor(bufOut, &output, sizeof(float));

    float expected = reduceRef<float>(ReduceSum, data);
    EXPECT_NEAR(output, expected, std::abs(expected) * 1e-4f + 1e-5f);
  }
}

// ============================================================================
// Prefix Scan Tests
// ============================================================================

class PrefixScanTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(PrefixScanTest, ExclusiveSum_Small) {
  // Small array: single workgroup
  for (uint32_t elements : {1u, 4u, 16u, 100u, 256u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    auto bufIn =
        runtime_->createTensor({elements}, DataType::Float32, data.data());
    auto bufOut = runtime_->createTensorEmpty({elements}, DataType::Float32);

    runtime_->encodeOperator(
        PrefixScanExclusiveSum,
        {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufOut, output.data(), elements * sizeof(float));

    // Verify exclusive prefix sum: output[i] = sum(input[0..i-1])
    float runningSum = 0.0f;
    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_NEAR(output[i], runningSum, std::abs(runningSum) * 1e-4f + 1e-5f)
          << "Mismatch at index " << i;
      runningSum += data[i];
    }
  }
}

TEST_F(PrefixScanTest, ExclusiveSum_Large) {
  // Large array: multi-workgroup (>256 elements)
  for (uint32_t elements : {257u, 1000u, 10000u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    auto bufIn =
        runtime_->createTensor({elements}, DataType::Float32, data.data());
    auto bufOut = runtime_->createTensorEmpty({elements}, DataType::Float32);

    runtime_->encodeOperator(
        PrefixScanExclusiveSum,
        {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufOut, output.data(), elements * sizeof(float));

    float runningSum = 0.0f;
    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_NEAR(output[i], runningSum, std::abs(runningSum) * 1e-3f + 1e-4f)
          << "Mismatch at index " << i;
      runningSum += data[i];
    }
  }
}

TEST_F(PrefixScanTest, InclusiveSum_Small) {
  for (uint32_t elements : {1u, 4u, 16u, 100u, 256u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    auto bufIn =
        runtime_->createTensor({elements}, DataType::Float32, data.data());
    auto bufOut = runtime_->createTensorEmpty({elements}, DataType::Float32);

    runtime_->encodeOperator(
        PrefixScanInclusiveSum,
        {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufOut, output.data(), elements * sizeof(float));

    // Verify inclusive prefix sum: output[i] = sum(input[0..i])
    float runningSum = 0.0f;
    for (uint32_t i = 0; i < elements; ++i) {
      runningSum += data[i];
      EXPECT_NEAR(output[i], runningSum, std::abs(runningSum) * 1e-4f + 1e-5f)
          << "Mismatch at index " << i;
    }
  }
}

TEST_F(PrefixScanTest, InclusiveSum_Large) {
  for (uint32_t elements : {257u, 1000u, 10000u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    auto bufIn =
        runtime_->createTensor({elements}, DataType::Float32, data.data());
    auto bufOut = runtime_->createTensorEmpty({elements}, DataType::Float32);

    runtime_->encodeOperator(
        PrefixScanInclusiveSum,
        {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufOut, output.data(), elements * sizeof(float));

    float runningSum = 0.0f;
    for (uint32_t i = 0; i < elements; ++i) {
      runningSum += data[i];
      EXPECT_NEAR(output[i], runningSum, std::abs(runningSum) * 1e-3f + 1e-4f)
          << "Mismatch at index " << i;
    }
  }
}

// ============================================================================
// Bitonic Sort Tests (Float32)
// ============================================================================

class BitonicSortTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(BitonicSortTest, Sort_SmallArrays) {
  for (uint32_t elements : {1u, 2u, 4u, 16u, 100u, 256u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);

    // Create indices [0, 1, 2, ..., N-1]
    std::vector<uint32_t> indices(elements);
    for (uint32_t i = 0; i < elements; ++i)
      indices[i] = i;

    auto bufKeys =
        runtime_->createTensor({elements}, DataType::Float32, data.data());
    auto bufVals =
        runtime_->createTensor({elements}, DataType::UInt32, indices.data());

    runtime_->encodeOperator(
        SortBitonic, {ComputeBinding(0, bufKeys), ComputeBinding(1, bufVals)});

    std::vector<float> sortedKeys(elements);
    std::vector<uint32_t> sortedVals(elements);
    runtime_->copyFromTensor(bufKeys, sortedKeys.data(),
                             elements * sizeof(float));
    runtime_->copyFromTensor(bufVals, sortedVals.data(),
                             elements * sizeof(uint32_t));

    // Verify ascending order
    for (uint32_t i = 1; i < elements; ++i) {
      EXPECT_LE(sortedKeys[i - 1], sortedKeys[i])
          << "Not sorted at index " << i;
    }

    // Verify indices are a valid permutation
    std::vector<uint32_t> sortedIndices(sortedVals.begin(), sortedVals.end());
    std::sort(sortedIndices.begin(), sortedIndices.end());
    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_EQ(sortedIndices[i], i) << "Invalid permutation at index " << i;
    }

    // Verify key-index correspondence
    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_EQ(sortedKeys[i], data[sortedVals[i]])
          << "Key-index mismatch at position " << i;
    }
  }
}

TEST_F(BitonicSortTest, Sort_LargeArray) {
  for (uint32_t elements : {1000u, 10000u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    std::vector<uint32_t> indices(elements);
    for (uint32_t i = 0; i < elements; ++i)
      indices[i] = i;

    auto bufKeys =
        runtime_->createTensor({elements}, DataType::Float32, data.data());
    auto bufVals =
        runtime_->createTensor({elements}, DataType::UInt32, indices.data());

    runtime_->encodeOperator(
        SortBitonic, {ComputeBinding(0, bufKeys), ComputeBinding(1, bufVals)});

    std::vector<float> sortedKeys(elements);
    std::vector<uint32_t> sortedVals(elements);
    runtime_->copyFromTensor(bufKeys, sortedKeys.data(),
                             elements * sizeof(float));
    runtime_->copyFromTensor(bufVals, sortedVals.data(),
                             elements * sizeof(uint32_t));

    // Verify ascending order
    for (uint32_t i = 1; i < elements; ++i) {
      EXPECT_LE(sortedKeys[i - 1], sortedKeys[i])
          << "Not sorted at index " << i;
    }

    // Verify indices are a valid permutation
    std::vector<uint32_t> sortedIndices(sortedVals.begin(), sortedVals.end());
    std::sort(sortedIndices.begin(), sortedIndices.end());
    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_EQ(sortedIndices[i], i);
    }
  }
}

TEST_F(BitonicSortTest, Sort_AlreadySorted) {
  uint32_t elements = 100;
  std::vector<float> data(elements);
  for (uint32_t i = 0; i < elements; ++i)
    data[i] = static_cast<float>(i);

  std::vector<uint32_t> indices(elements);
  for (uint32_t i = 0; i < elements; ++i)
    indices[i] = i;

  auto bufKeys =
      runtime_->createTensor({elements}, DataType::Float32, data.data());
  auto bufVals =
      runtime_->createTensor({elements}, DataType::UInt32, indices.data());

  runtime_->encodeOperator(
      SortBitonic, {ComputeBinding(0, bufKeys), ComputeBinding(1, bufVals)});

  std::vector<float> sortedKeys(elements);
  runtime_->copyFromTensor(bufKeys, sortedKeys.data(),
                           elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(sortedKeys[i], static_cast<float>(i));
  }
}

TEST_F(BitonicSortTest, Sort_ReverseSorted) {
  uint32_t elements = 100;
  std::vector<float> data(elements);
  for (uint32_t i = 0; i < elements; ++i)
    data[i] = static_cast<float>(elements - 1 - i);

  std::vector<uint32_t> indices(elements);
  for (uint32_t i = 0; i < elements; ++i)
    indices[i] = i;

  auto bufKeys =
      runtime_->createTensor({elements}, DataType::Float32, data.data());
  auto bufVals =
      runtime_->createTensor({elements}, DataType::UInt32, indices.data());

  runtime_->encodeOperator(
      SortBitonic, {ComputeBinding(0, bufKeys), ComputeBinding(1, bufVals)});

  std::vector<float> sortedKeys(elements);
  runtime_->copyFromTensor(bufKeys, sortedKeys.data(),
                           elements * sizeof(float));

  for (uint32_t i = 1; i < elements; ++i) {
    EXPECT_LE(sortedKeys[i - 1], sortedKeys[i]) << "Not sorted at index " << i;
  }
}

TEST_F(BitonicSortTest, Sort_AllSameValues) {
  uint32_t elements = 100;
  std::vector<float> data(elements, 5.0f);
  std::vector<uint32_t> indices(elements);
  for (uint32_t i = 0; i < elements; ++i)
    indices[i] = i;

  auto bufKeys =
      runtime_->createTensor({elements}, DataType::Float32, data.data());
  auto bufVals =
      runtime_->createTensor({elements}, DataType::UInt32, indices.data());

  runtime_->encodeOperator(
      SortBitonic, {ComputeBinding(0, bufKeys), ComputeBinding(1, bufVals)});

  std::vector<float> sortedKeys(elements);
  runtime_->copyFromTensor(bufKeys, sortedKeys.data(),
                           elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(sortedKeys[i], 5.0f);
  }
}

// ============================================================================
// Radix Sort Tests (UInt32)
// ============================================================================

class RadixSortTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(RadixSortTest, Sort_SmallArrays_UInt32) {
  for (uint32_t elements : {1u, 2u, 16u, 100u, 256u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<uint32_t>(elements, 42);
    std::vector<uint32_t> indices(elements);
    for (uint32_t i = 0; i < elements; ++i)
      indices[i] = i;

    auto bufKeys =
        runtime_->createTensor({elements}, DataType::UInt32, data.data());
    auto bufVals =
        runtime_->createTensor({elements}, DataType::UInt32, indices.data());

    runtime_->encodeOperator(
        SortRadix, {ComputeBinding(0, bufKeys), ComputeBinding(1, bufVals)});

    std::vector<uint32_t> sortedKeys(elements);
    std::vector<uint32_t> sortedVals(elements);
    runtime_->copyFromTensor(bufKeys, sortedKeys.data(),
                             elements * sizeof(uint32_t));
    runtime_->copyFromTensor(bufVals, sortedVals.data(),
                             elements * sizeof(uint32_t));

    // Verify ascending order
    for (uint32_t i = 1; i < elements; ++i) {
      EXPECT_LE(sortedKeys[i - 1], sortedKeys[i])
          << "Not sorted at index " << i;
    }

    // Verify indices are a valid permutation
    std::vector<uint32_t> sortedIndices(sortedVals.begin(), sortedVals.end());
    std::sort(sortedIndices.begin(), sortedIndices.end());
    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_EQ(sortedIndices[i], i) << "Invalid permutation at index " << i;
    }

    // Verify key-index correspondence
    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_EQ(sortedKeys[i], data[sortedVals[i]])
          << "Key-index mismatch at position " << i;
    }
  }
}

TEST_F(RadixSortTest, Sort_LargeArray_UInt32) {
  for (uint32_t elements : {1000u, 10000u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<uint32_t>(elements, 42);
    std::vector<uint32_t> indices(elements);
    for (uint32_t i = 0; i < elements; ++i)
      indices[i] = i;

    auto bufKeys =
        runtime_->createTensor({elements}, DataType::UInt32, data.data());
    auto bufVals =
        runtime_->createTensor({elements}, DataType::UInt32, indices.data());

    runtime_->encodeOperator(
        SortRadix, {ComputeBinding(0, bufKeys), ComputeBinding(1, bufVals)});

    std::vector<uint32_t> sortedKeys(elements);
    std::vector<uint32_t> sortedVals(elements);
    runtime_->copyFromTensor(bufKeys, sortedKeys.data(),
                             elements * sizeof(uint32_t));
    runtime_->copyFromTensor(bufVals, sortedVals.data(),
                             elements * sizeof(uint32_t));

    for (uint32_t i = 1; i < elements; ++i) {
      EXPECT_LE(sortedKeys[i - 1], sortedKeys[i])
          << "Not sorted at index " << i;
    }

    std::vector<uint32_t> sortedIndices(sortedVals.begin(), sortedVals.end());
    std::sort(sortedIndices.begin(), sortedIndices.end());
    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_EQ(sortedIndices[i], i);
    }
  }
}

TEST_F(RadixSortTest, Sort_AlreadySorted_UInt32) {
  uint32_t elements = 100;
  std::vector<uint32_t> data(elements);
  for (uint32_t i = 0; i < elements; ++i)
    data[i] = i;

  std::vector<uint32_t> indices(elements);
  for (uint32_t i = 0; i < elements; ++i)
    indices[i] = i;

  auto bufKeys =
      runtime_->createTensor({elements}, DataType::UInt32, data.data());
  auto bufVals =
      runtime_->createTensor({elements}, DataType::UInt32, indices.data());

  runtime_->encodeOperator(
      SortRadix, {ComputeBinding(0, bufKeys), ComputeBinding(1, bufVals)});

  std::vector<uint32_t> sortedKeys(elements);
  runtime_->copyFromTensor(bufKeys, sortedKeys.data(),
                           elements * sizeof(uint32_t));

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(sortedKeys[i], i);
  }
}

TEST_F(RadixSortTest, Sort_ReverseSorted_UInt32) {
  uint32_t elements = 100;
  std::vector<uint32_t> data(elements);
  for (uint32_t i = 0; i < elements; ++i)
    data[i] = elements - 1 - i;

  std::vector<uint32_t> indices(elements);
  for (uint32_t i = 0; i < elements; ++i)
    indices[i] = i;

  auto bufKeys =
      runtime_->createTensor({elements}, DataType::UInt32, data.data());
  auto bufVals =
      runtime_->createTensor({elements}, DataType::UInt32, indices.data());

  runtime_->encodeOperator(
      SortRadix, {ComputeBinding(0, bufKeys), ComputeBinding(1, bufVals)});

  std::vector<uint32_t> sortedKeys(elements);
  runtime_->copyFromTensor(bufKeys, sortedKeys.data(),
                           elements * sizeof(uint32_t));

  for (uint32_t i = 1; i < elements; ++i) {
    EXPECT_LE(sortedKeys[i - 1], sortedKeys[i]) << "Not sorted at index " << i;
  }
}

TEST_F(RadixSortTest, Sort_AllSameValues_UInt32) {
  uint32_t elements = 100;
  std::vector<uint32_t> data(elements, 42u);
  std::vector<uint32_t> indices(elements);
  for (uint32_t i = 0; i < elements; ++i)
    indices[i] = i;

  auto bufKeys =
      runtime_->createTensor({elements}, DataType::UInt32, data.data());
  auto bufVals =
      runtime_->createTensor({elements}, DataType::UInt32, indices.data());

  runtime_->encodeOperator(
      SortRadix, {ComputeBinding(0, bufKeys), ComputeBinding(1, bufVals)});

  std::vector<uint32_t> sortedKeys(elements);
  runtime_->copyFromTensor(bufKeys, sortedKeys.data(),
                           elements * sizeof(uint32_t));

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(sortedKeys[i], 42u);
  }
}

// ============================================================================
// Binary Vec-Scalar Tests - Int32 and UInt32
// ============================================================================

TEST_F(VulkanBackendTest, BinaryVecScalarOperators_Int32) {
  const DataType dtype = DataType::Int32;
  const int32_t scalar = 3;

  constexpr std::array<OperatorEnum, 20> kInt32BinaryVecScalarOps = {
      // Arithmetic
      BinaryVecScalarAdd, BinaryVecScalarSub, BinaryVecScalarMul,
      BinaryVecScalarDiv, BinaryVecScalarMod, BinaryVecScalarFloorDiv,
      // Comparison
      BinaryVecScalarEqual, BinaryVecScalarNotEqual, BinaryVecScalarLess,
      BinaryVecScalarLessEqual, BinaryVecScalarGreater,
      BinaryVecScalarGreaterEqual,
      // Min/Max
      BinaryVecScalarMin, BinaryVecScalarMax,
      // Bitwise
      BinaryVecScalarBitwiseAnd, BinaryVecScalarBitwiseOr,
      BinaryVecScalarBitwiseXor,
      // Logical
      BinaryVecScalarLogicalAnd, BinaryVecScalarLogicalOr,
      BinaryVecScalarLogicalXor};

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(int32_t);

      auto dataA = generateTestData<int32_t>(elements, 42);

      for (OperatorEnum op : kInt32BinaryVecScalarOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        // Pass scalar as float through DataReference (GPU reads as float)
        float scalarF = static_cast<float>(scalar);

        auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
        auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

        runtime_->encodeOperator(
            op, {ComputeBinding(0, bufferA), ComputeBinding(1, bufferOut),
                 ComputeBinding(2, DataReference(scalarF))});

        std::vector<int32_t> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          int32_t expected = binaryVecScalarRef(op, dataA[i], scalar);
          EXPECT_EQ(output[i], expected)
              << "Mismatch at index " << i << " for " << operatorName(op);
        }
      }
    }
  }
}

TEST_F(VulkanBackendTest, BinaryVecScalarOperators_UInt32) {
  const DataType dtype = DataType::UInt32;
  const uint32_t scalar = 3;

  constexpr std::array<OperatorEnum, 20> kUInt32BinaryVecScalarOps = {
      // Arithmetic
      BinaryVecScalarAdd, BinaryVecScalarSub, BinaryVecScalarMul,
      BinaryVecScalarDiv, BinaryVecScalarMod, BinaryVecScalarFloorDiv,
      // Comparison
      BinaryVecScalarEqual, BinaryVecScalarNotEqual, BinaryVecScalarLess,
      BinaryVecScalarLessEqual, BinaryVecScalarGreater,
      BinaryVecScalarGreaterEqual,
      // Min/Max
      BinaryVecScalarMin, BinaryVecScalarMax,
      // Bitwise
      BinaryVecScalarBitwiseAnd, BinaryVecScalarBitwiseOr,
      BinaryVecScalarBitwiseXor,
      // Logical
      BinaryVecScalarLogicalAnd, BinaryVecScalarLogicalOr,
      BinaryVecScalarLogicalXor};

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(uint32_t);

      auto dataA = generateTestData<uint32_t>(elements, 42);

      for (OperatorEnum op : kUInt32BinaryVecScalarOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        float scalarF = static_cast<float>(scalar);

        auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
        auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

        runtime_->encodeOperator(
            op, {ComputeBinding(0, bufferA), ComputeBinding(1, bufferOut),
                 ComputeBinding(2, DataReference(scalarF))});

        std::vector<uint32_t> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          uint32_t expected = binaryVecScalarRef(op, dataA[i], scalar);
          EXPECT_EQ(output[i], expected)
              << "Mismatch at index " << i << " for " << operatorName(op);
        }
      }
    }
  }
}

// ============================================================================
// Unary Operators - UInt32
// ============================================================================

TEST_F(VulkanBackendTest, UnaryOperators_UInt32) {
  const DataType dtype = DataType::UInt32;

  constexpr std::array<OperatorEnum, 8> kUInt32UnaryOps = {
      UnarySquare, UnaryReciprocal, UnarySign,       UnaryFloor,
      UnaryCeil,   UnaryRound,      UnaryLogicalNot, UnaryBitwiseNot};

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(uint32_t);

      auto dataIn = generateTestData<uint32_t>(elements, 42);

      for (OperatorEnum op : kUInt32UnaryOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
        auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

        runtime_->encodeOperator(
            op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

        std::vector<uint32_t> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          uint32_t expected = unaryRef(op, dataIn[i]);
          EXPECT_EQ(output[i], expected)
              << "Mismatch at index " << i << " for " << operatorName(op);
        }
      }
    }
  }
}

// ============================================================================
// Ternary Operators - Int32 and UInt32
// ============================================================================

TEST_F(VulkanBackendTest, TernaryClamp_Int32) {
  const DataType dtype = DataType::Int32;

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(int32_t);

      auto dataIn = generateTestData<int32_t>(elements, 42);
      float minVal = 20.0f;
      float maxVal = 80.0f;

      SCOPED_TRACE(std::string("Shape: ") + shapeToString(shape));

      auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
      auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

      float clampVals[2] = {minVal, maxVal};
      runtime_->encodeOperator(
          TernaryClamp,
          {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
           ComputeBinding(2, DataReference(clampVals, sizeof(clampVals)))});

      std::vector<int32_t> output(elements);
      runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

      for (uint32_t i = 0; i < elements; ++i) {
        int32_t expected =
            ternaryClampRef(dataIn[i], static_cast<int32_t>(minVal),
                            static_cast<int32_t>(maxVal));
        EXPECT_EQ(output[i], expected) << "Mismatch at index " << i;
      }
    }
  }
}

TEST_F(VulkanBackendTest, TernaryClamp_UInt32) {
  const DataType dtype = DataType::UInt32;

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(uint32_t);

      auto dataIn = generateTestData<uint32_t>(elements, 42);
      float minVal = 20.0f;
      float maxVal = 80.0f;

      SCOPED_TRACE(std::string("Shape: ") + shapeToString(shape));

      auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
      auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

      float clampVals[2] = {minVal, maxVal};
      runtime_->encodeOperator(
          TernaryClamp,
          {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut),
           ComputeBinding(2, DataReference(clampVals, sizeof(clampVals)))});

      std::vector<uint32_t> output(elements);
      runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

      for (uint32_t i = 0; i < elements; ++i) {
        uint32_t expected =
            ternaryClampRef(dataIn[i], static_cast<uint32_t>(minVal),
                            static_cast<uint32_t>(maxVal));
        EXPECT_EQ(output[i], expected) << "Mismatch at index " << i;
      }
    }
  }
}

TEST_F(VulkanBackendTest, TernarySelect_Int32) {
  const DataType dtype = DataType::Int32;

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(int32_t);

      auto dataCond = generateTestData<int32_t>(elements, 42);
      auto dataX = generateTestData<int32_t>(elements, 123);
      auto dataY = generateTestData<int32_t>(elements, 456);

      for (size_t i = 0; i < dataCond.size(); ++i) {
        dataCond[i] = (i % 3 == 0) ? 0 : dataCond[i];
      }

      SCOPED_TRACE(std::string("Shape: ") + shapeToString(shape));

      auto bufferCond = runtime_->createTensor(shape, dtype, dataCond.data());
      auto bufferX = runtime_->createTensor(shape, dtype, dataX.data());
      auto bufferY = runtime_->createTensor(shape, dtype, dataY.data());
      auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

      runtime_->encodeOperator(TernarySelect, {ComputeBinding(0, bufferCond),
                                               ComputeBinding(1, bufferX),
                                               ComputeBinding(2, bufferY),
                                               ComputeBinding(3, bufferOut)});

      std::vector<int32_t> output(elements);
      runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

      for (uint32_t i = 0; i < elements; ++i) {
        int32_t expected = ternarySelectRef(dataCond[i], dataX[i], dataY[i]);
        EXPECT_EQ(output[i], expected) << "Mismatch at index " << i;
      }
    }
  }
}

TEST_F(VulkanBackendTest, TernarySelect_UInt32) {
  const DataType dtype = DataType::UInt32;

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(uint32_t);

      auto dataCond = generateTestData<uint32_t>(elements, 42);
      auto dataX = generateTestData<uint32_t>(elements, 123);
      auto dataY = generateTestData<uint32_t>(elements, 456);

      for (size_t i = 0; i < dataCond.size(); ++i) {
        dataCond[i] = (i % 3 == 0) ? 0u : dataCond[i];
      }

      SCOPED_TRACE(std::string("Shape: ") + shapeToString(shape));

      auto bufferCond = runtime_->createTensor(shape, dtype, dataCond.data());
      auto bufferX = runtime_->createTensor(shape, dtype, dataX.data());
      auto bufferY = runtime_->createTensor(shape, dtype, dataY.data());
      auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

      runtime_->encodeOperator(TernarySelect, {ComputeBinding(0, bufferCond),
                                               ComputeBinding(1, bufferX),
                                               ComputeBinding(2, bufferY),
                                               ComputeBinding(3, bufferOut)});

      std::vector<uint32_t> output(elements);
      runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

      for (uint32_t i = 0; i < elements; ++i) {
        uint32_t expected = ternarySelectRef(dataCond[i], dataX[i], dataY[i]);
        EXPECT_EQ(output[i], expected) << "Mismatch at index " << i;
      }
    }
  }
}

// ============================================================================
// Reduction Operators - Int32 and UInt32
// ============================================================================

TEST_F(VulkanBackendTest, ReductionOperators_Int32) {
  const DataType dtype = DataType::Int32;

  // Int32 supports all reduction ops except Mean (integer division)
  constexpr std::array<OperatorEnum, 6> kInt32ReductionOps = {
      ReduceSum, ReduceMin, ReduceMax, ReduceProd, ReduceAny, ReduceAll};

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);

      // Use small values to avoid overflow with ReduceProd
      std::mt19937 gen(42);
      std::uniform_int_distribution<int32_t> dist(1, 3);
      std::vector<int32_t> dataIn(elements);
      for (auto &v : dataIn)
        v = dist(gen);

      for (OperatorEnum op : kInt32ReductionOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
        auto bufferOut = runtime_->createTensorEmpty({1}, dtype);

        int32_t initVal = 0;
        if (op == ReduceProd || op == ReduceAll)
          initVal = 1;
        else if (op == ReduceMin)
          initVal = std::numeric_limits<int32_t>::max();
        else if (op == ReduceMax)
          initVal = std::numeric_limits<int32_t>::lowest();
        runtime_->copyToTensor(bufferOut, &initVal, sizeof(int32_t));

        runtime_->encodeOperator(
            op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

        int32_t output = 0;
        runtime_->copyFromTensor(bufferOut, &output, sizeof(int32_t));

        int32_t expected = reduceRef(op, dataIn);
        EXPECT_EQ(output, expected) << "Mismatch for " << operatorName(op);
      }
    }
  }
}

TEST_F(VulkanBackendTest, ReductionOperators_UInt32) {
  const DataType dtype = DataType::UInt32;

  constexpr std::array<OperatorEnum, 6> kUInt32ReductionOps = {
      ReduceSum, ReduceMin, ReduceMax, ReduceProd, ReduceAny, ReduceAll};

  for (size_t numDims : kDimensionCounts) {
    for (const auto &shape : generateShapes(numDims)) {
      const uint32_t elements = totalElements(shape);

      std::mt19937 gen(42);
      std::uniform_int_distribution<uint32_t> dist(1, 3);
      std::vector<uint32_t> dataIn(elements);
      for (auto &v : dataIn)
        v = dist(gen);

      for (OperatorEnum op : kUInt32ReductionOps) {
        SCOPED_TRACE(std::string("Op: ") + operatorName(op) +
                     " Shape: " + shapeToString(shape));

        auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());
        auto bufferOut = runtime_->createTensorEmpty({1}, dtype);

        uint32_t initVal = 0;
        if (op == ReduceProd || op == ReduceAll)
          initVal = 1;
        else if (op == ReduceMin)
          initVal = std::numeric_limits<uint32_t>::max();
        else if (op == ReduceMax)
          initVal = std::numeric_limits<uint32_t>::lowest();
        runtime_->copyToTensor(bufferOut, &initVal, sizeof(uint32_t));

        runtime_->encodeOperator(
            op, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

        uint32_t output = 0;
        runtime_->copyFromTensor(bufferOut, &output, sizeof(uint32_t));

        uint32_t expected = reduceRef(op, dataIn);
        EXPECT_EQ(output, expected) << "Mismatch for " << operatorName(op);
      }
    }
  }
}

// ============================================================================
// Matrix Operation Tests
// ============================================================================

class MatrixOpsTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(MatrixOpsTest, MatMul_Square) {
  const DataType dtype = DataType::Float32;

  // 4x4 * 4x4 = 4x4
  const uint32_t M = 4, K = 4, N = 4;
  std::vector<float> A = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> B = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  // A * I = A

  auto bufA = runtime_->createTensor({M, K}, dtype, A.data());
  auto bufB = runtime_->createTensor({K, N}, dtype, B.data());
  auto bufC = runtime_->createTensorEmpty({M, N}, dtype);

  uint32_t dims[3] = {M, K, N};
  runtime_->encodeOperator(
      MatMul, {ComputeBinding(0, bufA), ComputeBinding(1, bufB),
               ComputeBinding(2, bufC),
               ComputeBinding(3, DataReference(dims, sizeof(dims)))});

  std::vector<float> output(M * N);
  runtime_->copyFromTensor(bufC, output.data(), M * N * sizeof(float));

  for (uint32_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(output[i], A[i], 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(MatrixOpsTest, MatMul_Rectangular) {
  const DataType dtype = DataType::Float32;

  // 2x4 * 4x8 = 2x8
  const uint32_t M = 2, K = 4, N = 8;
  auto dataA = generateTestData<float>(M * K, 42);
  auto dataB = generateTestData<float>(K * N, 123);

  auto bufA = runtime_->createTensor({M, K}, dtype, dataA.data());
  auto bufB = runtime_->createTensor({K, N}, dtype, dataB.data());
  auto bufC = runtime_->createTensorEmpty({M, N}, dtype);

  uint32_t dims[3] = {M, K, N};
  runtime_->encodeOperator(
      MatMul, {ComputeBinding(0, bufA), ComputeBinding(1, bufB),
               ComputeBinding(2, bufC),
               ComputeBinding(3, DataReference(dims, sizeof(dims)))});

  std::vector<float> output(M * N);
  runtime_->copyFromTensor(bufC, output.data(), M * N * sizeof(float));

  // Reference matmul
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float expected = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        expected += dataA[i * K + k] * dataB[k * N + j];
      }
      EXPECT_NEAR(output[i * N + j], expected,
                  std::abs(expected) * 1e-4f + 1e-5f)
          << "Mismatch at [" << i << ", " << j << "]";
    }
  }
}

TEST_F(MatrixOpsTest, MatMul_LargerMatrices) {
  const DataType dtype = DataType::Float32;

  // Test several sizes including non-multiples of tile size (16)
  struct TestCase {
    uint32_t M, K, N;
  };
  std::array<TestCase, 4> testCases = {
      {{8, 8, 8}, {16, 16, 16}, {7, 12, 4}, {4, 4, 16}}};

  for (const auto &tc : testCases) {
    SCOPED_TRACE("MatMul [" + std::to_string(tc.M) + "x" +
                 std::to_string(tc.K) + "] * [" + std::to_string(tc.K) + "x" +
                 std::to_string(tc.N) + "]");

    auto dataA = generateTestData<float>(tc.M * tc.K, 42);
    auto dataB = generateTestData<float>(tc.K * tc.N, 123);

    auto bufA = runtime_->createTensor({tc.M, tc.K}, dtype, dataA.data());
    auto bufB = runtime_->createTensor({tc.K, tc.N}, dtype, dataB.data());
    auto bufC = runtime_->createTensorEmpty({tc.M, tc.N}, dtype);

    uint32_t dims[3] = {tc.M, tc.K, tc.N};
    runtime_->encodeOperator(
        MatMul, {ComputeBinding(0, bufA), ComputeBinding(1, bufB),
                 ComputeBinding(2, bufC),
                 ComputeBinding(3, DataReference(dims, sizeof(dims)))});

    std::vector<float> output(tc.M * tc.N);
    runtime_->copyFromTensor(bufC, output.data(), tc.M * tc.N * sizeof(float));

    for (uint32_t i = 0; i < tc.M; ++i) {
      for (uint32_t j = 0; j < tc.N; ++j) {
        float expected = 0.0f;
        for (uint32_t k = 0; k < tc.K; ++k) {
          expected += dataA[i * tc.K + k] * dataB[k * tc.N + j];
        }
        EXPECT_NEAR(output[i * tc.N + j], expected,
                    std::abs(expected) * 1e-4f + 1e-5f)
            << "Mismatch at [" << i << ", " << j << "]";
      }
    }
  }
}

TEST_F(MatrixOpsTest, Transpose_Square) {
  const DataType dtype = DataType::Float32;
  const uint32_t M = 4, N = 4;

  std::vector<float> data = {1, 2,  3,  4,  5,  6,  7,  8,
                             9, 10, 11, 12, 13, 14, 15, 16};

  auto bufIn = runtime_->createTensor({M, N}, dtype, data.data());
  auto bufOut = runtime_->createTensorEmpty({N, M}, dtype);

  uint32_t dims[3] = {M, 0, N}; // M and N dimensions
  runtime_->encodeOperator(
      Transpose, {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut),
                  ComputeBinding(2, DataReference(dims, sizeof(dims)))});

  std::vector<float> output(M * N);
  runtime_->copyFromTensor(bufOut, output.data(), M * N * sizeof(float));

  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      EXPECT_NEAR(output[j * M + i], data[i * N + j], 1e-5f)
          << "Mismatch at [" << j << ", " << i << "]";
    }
  }
}

TEST_F(MatrixOpsTest, Transpose_Rectangular) {
  const DataType dtype = DataType::Float32;

  struct TestCase {
    uint32_t M, N;
  };
  std::array<TestCase, 3> testCases = {{{3, 4}, {4, 8}, {7, 12}}};

  for (const auto &tc : testCases) {
    SCOPED_TRACE("Transpose [" + std::to_string(tc.M) + "x" +
                 std::to_string(tc.N) + "]");

    auto dataIn = generateTestData<float>(tc.M * tc.N, 42);

    auto bufIn = runtime_->createTensor({tc.M, tc.N}, dtype, dataIn.data());
    auto bufOut = runtime_->createTensorEmpty({tc.N, tc.M}, dtype);

    uint32_t dims[3] = {tc.M, 0, tc.N};
    runtime_->encodeOperator(
        Transpose, {ComputeBinding(0, bufIn), ComputeBinding(1, bufOut),
                    ComputeBinding(2, DataReference(dims, sizeof(dims)))});

    std::vector<float> output(tc.M * tc.N);
    runtime_->copyFromTensor(bufOut, output.data(),
                             tc.M * tc.N * sizeof(float));

    for (uint32_t i = 0; i < tc.M; ++i) {
      for (uint32_t j = 0; j < tc.N; ++j) {
        EXPECT_NEAR(output[j * tc.M + i], dataIn[i * tc.N + j], 1e-5f)
            << "Mismatch at [" << j << ", " << i << "]";
      }
    }
  }
}

TEST_F(MatrixOpsTest, Dot_Basic) {
  const DataType dtype = DataType::Float32;

  std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> dataB = {5.0f, 6.0f, 7.0f, 8.0f};
  const uint32_t elements = 4;

  auto bufA = runtime_->createTensor({elements}, dtype, dataA.data());
  auto bufB = runtime_->createTensor({elements}, dtype, dataB.data());
  auto bufOut = runtime_->createTensorEmpty({1}, dtype);

  // Initialize output to 0
  float initVal = 0.0f;
  runtime_->copyToTensor(bufOut, &initVal, sizeof(float));

  uint32_t dims[3] = {elements, 0, 0};
  runtime_->encodeOperator(
      Dot, {ComputeBinding(0, bufA), ComputeBinding(1, bufB),
            ComputeBinding(2, bufOut),
            ComputeBinding(3, DataReference(dims, sizeof(dims)))});

  float output = 0.0f;
  runtime_->copyFromTensor(bufOut, &output, sizeof(float));

  float expected = 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8; // = 70
  EXPECT_NEAR(output, expected, 1e-4f);
}

TEST_F(MatrixOpsTest, Dot_LargerVectors) {
  const DataType dtype = DataType::Float32;

  for (uint32_t elements : {8u, 16u, 100u, 256u, 1024u}) {
    SCOPED_TRACE("Dot elements=" + std::to_string(elements));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufA = runtime_->createTensor({elements}, dtype, dataA.data());
    auto bufB = runtime_->createTensor({elements}, dtype, dataB.data());
    auto bufOut = runtime_->createTensorEmpty({1}, dtype);

    float initVal = 0.0f;
    runtime_->copyToTensor(bufOut, &initVal, sizeof(float));

    uint32_t dims[3] = {elements, 0, 0};
    runtime_->encodeOperator(
        Dot, {ComputeBinding(0, bufA), ComputeBinding(1, bufB),
              ComputeBinding(2, bufOut),
              ComputeBinding(3, DataReference(dims, sizeof(dims)))});

    float output = 0.0f;
    runtime_->copyFromTensor(bufOut, &output, sizeof(float));

    double expected = 0.0;
    for (uint32_t i = 0; i < elements; ++i) {
      expected += static_cast<double>(dataA[i]) * static_cast<double>(dataB[i]);
    }
    EXPECT_NEAR(output, static_cast<float>(expected),
                std::abs(static_cast<float>(expected)) * 1e-3f + 1e-4f);
  }
}

// ============================================================================
// Global Norm Test
// ============================================================================

class NormTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(NormTest, Norm_KnownValues) {
  const DataType dtype = DataType::Float32;

  // 3-4-5 triangle: sqrt(3^2 + 4^2) = 5
  std::vector<float> data = {3.0f, 4.0f, 0.0f, 0.0f};
  const uint32_t elements = 4;

  auto bufferIn = runtime_->createTensor({elements}, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty({1}, dtype);

  float initVal = 0.0f;
  runtime_->copyToTensor(bufferOut, &initVal, sizeof(float));

  runtime_->encodeOperator(
      Norm, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

  float output = 0.0f;
  runtime_->copyFromTensor(bufferOut, &output, sizeof(float));

  EXPECT_NEAR(output, 5.0f, 1e-4f);
}

TEST_F(NormTest, Norm_VariousSizes) {
  const DataType dtype = DataType::Float32;

  for (uint32_t elements : {4u, 8u, 16u, 100u, 256u, 1024u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);

    auto bufferIn = runtime_->createTensor({elements}, dtype, data.data());
    auto bufferOut = runtime_->createTensorEmpty({1}, dtype);

    float initVal = 0.0f;
    runtime_->copyToTensor(bufferOut, &initVal, sizeof(float));

    runtime_->encodeOperator(
        Norm, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

    float output = 0.0f;
    runtime_->copyFromTensor(bufferOut, &output, sizeof(float));

    // Reference: L2 norm
    double sumSq = 0.0;
    for (uint32_t i = 0; i < elements; ++i) {
      sumSq += static_cast<double>(data[i]) * static_cast<double>(data[i]);
    }
    float expected = static_cast<float>(std::sqrt(sumSq));

    EXPECT_NEAR(output, expected, std::abs(expected) * 1e-3f + 1e-4f);
  }
}

TEST_F(NormTest, Norm_MultiDimensional) {
  const DataType dtype = DataType::Float32;

  // 2D tensor
  std::vector<uint32_t> shape = {3, 4};
  const uint32_t elements = totalElements(shape);
  auto data = generateTestData<float>(elements, 42);

  auto bufferIn = runtime_->createTensor(shape, dtype, data.data());
  auto bufferOut = runtime_->createTensorEmpty({1}, dtype);

  float initVal = 0.0f;
  runtime_->copyToTensor(bufferOut, &initVal, sizeof(float));

  runtime_->encodeOperator(
      Norm, {ComputeBinding(0, bufferIn), ComputeBinding(1, bufferOut)});

  float output = 0.0f;
  runtime_->copyFromTensor(bufferOut, &output, sizeof(float));

  double sumSq = 0.0;
  for (uint32_t i = 0; i < elements; ++i) {
    sumSq += static_cast<double>(data[i]) * static_cast<double>(data[i]);
  }
  float expected = static_cast<float>(std::sqrt(sumSq));

  EXPECT_NEAR(output, expected, std::abs(expected) * 1e-3f + 1e-4f);
}

// ============================================================================
// Tensor Creation Operation Tests
// ============================================================================

class TensorCreationTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(TensorCreationTest, Zeros_Float32) {
  const DataType dtype = DataType::Float32;

  for (uint32_t elements : {4u, 8u, 16u, 100u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

    runtime_->encodeOperator(Zeros, {ComputeBinding(0, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(),
                             elements * sizeof(float));

    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_EQ(output[i], 0.0f) << "Mismatch at index " << i;
    }
  }
}

TEST_F(TensorCreationTest, Ones_Float32) {
  const DataType dtype = DataType::Float32;

  for (uint32_t elements : {4u, 8u, 16u, 100u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

    runtime_->encodeOperator(Ones, {ComputeBinding(0, bufferOut)});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(),
                             elements * sizeof(float));

    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_EQ(output[i], 1.0f) << "Mismatch at index " << i;
    }
  }
}

TEST_F(TensorCreationTest, Full_Float32) {
  const DataType dtype = DataType::Float32;
  const float fillValue = 3.14f;

  for (uint32_t elements : {4u, 8u, 16u, 100u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

    runtime_->encodeOperator(Full,
                             {ComputeBinding(0, bufferOut),
                              ComputeBinding(1, DataReference(fillValue))});

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(),
                             elements * sizeof(float));

    for (uint32_t i = 0; i < elements; ++i) {
      EXPECT_NEAR(output[i], fillValue, 1e-5f) << "Mismatch at index " << i;
    }
  }
}

TEST_F(TensorCreationTest, Arange_Float32) {
  const DataType dtype = DataType::Float32;

  // arange(0, 8, 1) -> [0, 1, 2, 3, 4, 5, 6, 7]
  const uint32_t elements = 8;
  const float start = 0.0f;
  const float step = 1.0f;

  auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

  float params[2] = {start, step};
  runtime_->encodeOperator(
      Arange, {ComputeBinding(0, bufferOut),
               ComputeBinding(1, DataReference(params, sizeof(params)))});

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = start + static_cast<float>(i) * step;
    EXPECT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Arange_WithStep) {
  const DataType dtype = DataType::Float32;

  // arange(1, ?, 0.5) -> [1.0, 1.5, 2.0, 2.5, ...]
  const uint32_t elements = 8;
  const float start = 1.0f;
  const float step = 0.5f;

  auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

  float params[2] = {start, step};
  runtime_->encodeOperator(
      Arange, {ComputeBinding(0, bufferOut),
               ComputeBinding(1, DataReference(params, sizeof(params)))});

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = start + static_cast<float>(i) * step;
    EXPECT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Linspace_Float32) {
  const DataType dtype = DataType::Float32;

  // linspace(0, 1, 5) -> [0.0, 0.25, 0.5, 0.75, 1.0]
  const uint32_t elements = 8;
  const float start = 0.0f;
  const float end = 1.0f;
  const float step = (end - start) / static_cast<float>(elements - 1);

  auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

  float params[2] = {start, step};
  runtime_->encodeOperator(
      Linspace, {ComputeBinding(0, bufferOut),
                 ComputeBinding(1, DataReference(params, sizeof(params)))});

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = start + static_cast<float>(i) * step;
    EXPECT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Zeros_MultiDimensional) {
  const DataType dtype = DataType::Float32;

  std::vector<uint32_t> shape = {3, 4};
  const uint32_t elements = totalElements(shape);

  auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

  runtime_->encodeOperator(Zeros, {ComputeBinding(0, bufferOut)});

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(output[i], 0.0f) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Ones_MultiDimensional) {
  const DataType dtype = DataType::Float32;

  std::vector<uint32_t> shape = {3, 4};
  const uint32_t elements = totalElements(shape);

  auto bufferOut = runtime_->createTensorEmpty(shape, dtype);

  runtime_->encodeOperator(Ones, {ComputeBinding(0, bufferOut)});

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(output[i], 1.0f) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Zeros_Int32) {
  const DataType dtype = DataType::Int32;

  const uint32_t elements = 16;
  auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

  runtime_->encodeOperator(Zeros, {ComputeBinding(0, bufferOut)});

  std::vector<int32_t> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(),
                           elements * sizeof(int32_t));

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(output[i], 0) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Ones_UInt32) {
  const DataType dtype = DataType::UInt32;

  const uint32_t elements = 16;
  auto bufferOut = runtime_->createTensorEmpty({elements}, dtype);

  runtime_->encodeOperator(Ones, {ComputeBinding(0, bufferOut)});

  std::vector<uint32_t> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(),
                           elements * sizeof(uint32_t));

  for (uint32_t i = 0; i < elements; ++i) {
    EXPECT_EQ(output[i], 1u) << "Mismatch at index " << i;
  }
}

} // namespace
} // namespace cut
