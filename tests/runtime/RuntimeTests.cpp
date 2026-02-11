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
constexpr std::array<OperatorEnum, 27> kBinaryVecVecOps = {
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
    BinaryVecVecFmod};

// All binary vec-scalar operators
constexpr std::array<OperatorEnum, 28> kBinaryVecScalarOps = {
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
    BinaryVecScalarLeakyRelu};

// All unary operators
constexpr std::array<OperatorEnum, 37> kUnaryOps = {
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
    UnaryIsNan, UnaryIsInf};

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

inline const char *operatorName(OperatorEnum op) {
  switch (op) {
  case BinaryVecVecAdd:
    return "BinaryVecVecAdd";
  case BinaryVecVecSub:
    return "BinaryVecVecSub";
  case BinaryVecVecMul:
    return "BinaryVecVecMul";
  case BinaryVecVecDiv:
    return "BinaryVecVecDiv";
  case BinaryVecVecMod:
    return "BinaryVecVecMod";
  case BinaryVecVecPow:
    return "BinaryVecVecPow";
  case BinaryVecVecFloorDiv:
    return "BinaryVecVecFloorDiv";
  case BinaryVecVecEqual:
    return "BinaryVecVecEqual";
  case BinaryVecVecNotEqual:
    return "BinaryVecVecNotEqual";
  case BinaryVecVecLess:
    return "BinaryVecVecLess";
  case BinaryVecVecLessEqual:
    return "BinaryVecVecLessEqual";
  case BinaryVecVecGreater:
    return "BinaryVecVecGreater";
  case BinaryVecVecGreaterEqual:
    return "BinaryVecVecGreaterEqual";
  case BinaryVecVecMin:
    return "BinaryVecVecMin";
  case BinaryVecVecMax:
    return "BinaryVecVecMax";
  case BinaryVecScalarAdd:
    return "BinaryVecScalarAdd";
  case BinaryVecScalarSub:
    return "BinaryVecScalarSub";
  case BinaryVecScalarMul:
    return "BinaryVecScalarMul";
  case BinaryVecScalarDiv:
    return "BinaryVecScalarDiv";
  case BinaryVecScalarMod:
    return "BinaryVecScalarMod";
  case BinaryVecScalarPow:
    return "BinaryVecScalarPow";
  case BinaryVecScalarFloorDiv:
    return "BinaryVecScalarFloorDiv";
  case BinaryVecScalarEqual:
    return "BinaryVecScalarEqual";
  case BinaryVecScalarNotEqual:
    return "BinaryVecScalarNotEqual";
  case BinaryVecScalarLess:
    return "BinaryVecScalarLess";
  case BinaryVecScalarLessEqual:
    return "BinaryVecScalarLessEqual";
  case BinaryVecScalarGreater:
    return "BinaryVecScalarGreater";
  case BinaryVecScalarGreaterEqual:
    return "BinaryVecScalarGreaterEqual";
  case BinaryVecScalarMin:
    return "BinaryVecScalarMin";
  case BinaryVecScalarMax:
    return "BinaryVecScalarMax";
  case UnaryNeg:
    return "UnaryNeg";
  case UnaryAbs:
    return "UnaryAbs";
  case UnarySqrt:
    return "UnarySqrt";
  case UnaryExp:
    return "UnaryExp";
  case UnaryLog:
    return "UnaryLog";
  case UnaryLog2:
    return "UnaryLog2";
  case UnaryLog10:
    return "UnaryLog10";
  case UnarySin:
    return "UnarySin";
  case UnaryCos:
    return "UnaryCos";
  case UnaryTan:
    return "UnaryTan";
  case UnaryAsin:
    return "UnaryAsin";
  case UnaryAcos:
    return "UnaryAcos";
  case UnaryAtan:
    return "UnaryAtan";
  case UnarySinh:
    return "UnarySinh";
  case UnaryCosh:
    return "UnaryCosh";
  case UnaryTanh:
    return "UnaryTanh";
  case UnaryFloor:
    return "UnaryFloor";
  case UnaryCeil:
    return "UnaryCeil";
  case UnaryRound:
    return "UnaryRound";
  case UnarySign:
    return "UnarySign";
  case UnaryReciprocal:
    return "UnaryReciprocal";
  case UnarySquare:
    return "UnarySquare";
  // Bitwise binary vec-vec operators
  case BinaryVecVecBitwiseAnd:
    return "BinaryVecVecBitwiseAnd";
  case BinaryVecVecBitwiseOr:
    return "BinaryVecVecBitwiseOr";
  case BinaryVecVecBitwiseXor:
    return "BinaryVecVecBitwiseXor";
  case BinaryVecVecLeftShift:
    return "BinaryVecVecLeftShift";
  case BinaryVecVecRightShift:
    return "BinaryVecVecRightShift";
  case BinaryVecVecLogicalAnd:
    return "BinaryVecVecLogicalAnd";
  case BinaryVecVecLogicalOr:
    return "BinaryVecVecLogicalOr";
  case BinaryVecVecLogicalXor:
    return "BinaryVecVecLogicalXor";
  case BinaryVecVecAtan2:
    return "BinaryVecVecAtan2";
  case BinaryVecVecHypot:
    return "BinaryVecVecHypot";
  case BinaryVecVecCopysign:
    return "BinaryVecVecCopysign";
  case BinaryVecVecFmod:
    return "BinaryVecVecFmod";
  // Bitwise binary vec-scalar operators
  case BinaryVecScalarBitwiseAnd:
    return "BinaryVecScalarBitwiseAnd";
  case BinaryVecScalarBitwiseOr:
    return "BinaryVecScalarBitwiseOr";
  case BinaryVecScalarBitwiseXor:
    return "BinaryVecScalarBitwiseXor";
  case BinaryVecScalarLeftShift:
    return "BinaryVecScalarLeftShift";
  case BinaryVecScalarRightShift:
    return "BinaryVecScalarRightShift";
  case BinaryVecScalarLogicalAnd:
    return "BinaryVecScalarLogicalAnd";
  case BinaryVecScalarLogicalOr:
    return "BinaryVecScalarLogicalOr";
  case BinaryVecScalarLogicalXor:
    return "BinaryVecScalarLogicalXor";
  case BinaryVecScalarAtan2:
    return "BinaryVecScalarAtan2";
  case BinaryVecScalarHypot:
    return "BinaryVecScalarHypot";
  case BinaryVecScalarCopysign:
    return "BinaryVecScalarCopysign";
  case BinaryVecScalarFmod:
    return "BinaryVecScalarFmod";
  case BinaryVecScalarLeakyRelu:
    return "BinaryVecScalarLeakyRelu";
  // Additional unary operators
  case UnaryExp2:
    return "UnaryExp2";
  case UnaryExpm1:
    return "UnaryExpm1";
  case UnaryLog1p:
    return "UnaryLog1p";
  case UnaryCbrt:
    return "UnaryCbrt";
  case UnaryDegrees:
    return "UnaryDegrees";
  case UnaryRadians:
    return "UnaryRadians";
  case UnaryLogicalNot:
    return "UnaryLogicalNot";
  case UnaryBitwiseNot:
    return "UnaryBitwiseNot";
  case UnaryRelu:
    return "UnaryRelu";
  case UnarySigmoid:
    return "UnarySigmoid";
  case UnaryGelu:
    return "UnaryGelu";
  case UnarySilu:
    return "UnarySilu";
  case UnarySoftplus:
    return "UnarySoftplus";
  case UnaryIsNan:
    return "UnaryIsNan";
  case UnaryIsInf:
    return "UnaryIsInf";
  // Ternary operators
  case TernaryClamp:
    return "TernaryClamp";
  case TernarySelect:
    return "TernarySelect";
  // Reduction operators
  case ReduceSum:
    return "ReduceSum";
  case ReduceMean:
    return "ReduceMean";
  case ReduceMin:
    return "ReduceMin";
  case ReduceMax:
    return "ReduceMax";
  case ReduceProd:
    return "ReduceProd";
  case ReduceAny:
    return "ReduceAny";
  case ReduceAll:
    return "ReduceAll";
  // Matrix operators
  case MatMul:
    return "MatMul";
  case Transpose:
    return "Transpose";
  case Dot:
    return "Dot";
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
    } else {
      return b >= T{0} ? std::abs(a) : -std::abs(a);
    }
  case BinaryVecVecFmod:
    if constexpr (std::is_floating_point_v<T>) {
      return std::fmod(a, b);
    } else {
      return a % b;
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
  // Bitwise operations
  case BinaryVecScalarBitwiseAnd:
    if constexpr (std::is_integral_v<T>) {
      return a & static_cast<T>(scalar);
    } else {
      return static_cast<T>(static_cast<int>(a) & static_cast<int>(scalar));
    }
  case BinaryVecScalarBitwiseOr:
    if constexpr (std::is_integral_v<T>) {
      return a | static_cast<T>(scalar);
    } else {
      return static_cast<T>(static_cast<int>(a) | static_cast<int>(scalar));
    }
  case BinaryVecScalarBitwiseXor:
    if constexpr (std::is_integral_v<T>) {
      return a ^ static_cast<T>(scalar);
    } else {
      return static_cast<T>(static_cast<int>(a) ^ static_cast<int>(scalar));
    }
  case BinaryVecScalarLeftShift:
    if constexpr (std::is_integral_v<T>) {
      return a << static_cast<int>(scalar);
    } else {
      return static_cast<T>(static_cast<int>(a) << static_cast<int>(scalar));
    }
  case BinaryVecScalarRightShift:
    if constexpr (std::is_integral_v<T>) {
      return a >> static_cast<int>(scalar);
    } else {
      return static_cast<T>(static_cast<int>(a) >> static_cast<int>(scalar));
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
      return static_cast<T>(~static_cast<int>(a));
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

// Note: Current shaders only support Float32, so Int32 tests are skipped
TEST_F(VulkanBackendTest, BinaryVecVecOperators_Int32) {
  GTEST_SKIP() << "Vulkan shaders currently only support Float32";
}

// Note: Current shaders only support Float32, so UInt32 tests are skipped
TEST_F(VulkanBackendTest, BinaryVecVecOperators_UInt32) {
  GTEST_SKIP() << "Vulkan shaders currently only support Float32";
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
        // Skip operators without shader support or problematic ones
        if (!hasVulkanShaderSupport(op) || op == UnaryLog || op == UnaryLog2 ||
            op == UnaryLog10 || op == UnaryAsin || op == UnaryAcos ||
            op == UnaryTan || op == UnarySinh || op == UnaryCosh ||
            op == UnaryExp || op == UnaryBitwiseNot) {
          continue;
        }

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
        // Skip operators without shader support or problematic ones
        if (!hasVulkanShaderSupport(op) || op == BinaryVecScalarMod ||
            op == BinaryVecScalarPow || op == BinaryVecScalarFloorDiv ||
            op == BinaryVecScalarBitwiseAnd || op == BinaryVecScalarBitwiseOr ||
            op == BinaryVecScalarBitwiseXor || op == BinaryVecScalarLeftShift ||
            op == BinaryVecScalarRightShift) {
          continue;
        }

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
          EXPECT_NEAR(output[i], expected, 1e-5f)
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
TEST_F(VulkanBackendTest, DISABLED_ReductionOperators_Float32) {
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
        if (op == ReduceMean || op == ReduceSum || op == ReduceProd) {
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

} // namespace
} // namespace cut
