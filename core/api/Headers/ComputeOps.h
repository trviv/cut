#pragma once

#include <cstdint>

// Include shared definitions (compatible with GLSL)
#include "ComputeOpsShared.h"

namespace cut {

/**
 * Operator enum for built-in compute operations.
 * Used by both GPU shaders and CPU kernels.
 * Values match OP_* defines in ComputeOpsShared.h for shader compatibility.
 */
enum OperatorEnum {
  // ===========================================================================
  // Binary vec-vec operations (0-29)
  // ===========================================================================

  // Arithmetic
  BinaryVecVecAdd = OP_BINARY_VEC_VEC_ADD,
  BinaryVecVecSub = OP_BINARY_VEC_VEC_SUB,
  BinaryVecVecMul = OP_BINARY_VEC_VEC_MUL,
  BinaryVecVecDiv = OP_BINARY_VEC_VEC_DIV,
  BinaryVecVecMod = OP_BINARY_VEC_VEC_MOD,
  BinaryVecVecPow = OP_BINARY_VEC_VEC_POW,
  BinaryVecVecFloorDiv = OP_BINARY_VEC_VEC_FLOOR_DIV,

  // Comparison
  BinaryVecVecEqual = OP_BINARY_VEC_VEC_EQUAL,
  BinaryVecVecNotEqual = OP_BINARY_VEC_VEC_NOT_EQUAL,
  BinaryVecVecLess = OP_BINARY_VEC_VEC_LESS,
  BinaryVecVecLessEqual = OP_BINARY_VEC_VEC_LESS_EQUAL,
  BinaryVecVecGreater = OP_BINARY_VEC_VEC_GREATER,
  BinaryVecVecGreaterEqual = OP_BINARY_VEC_VEC_GREATER_EQUAL,

  // Min/Max
  BinaryVecVecMin = OP_BINARY_VEC_VEC_MIN,
  BinaryVecVecMax = OP_BINARY_VEC_VEC_MAX,

  // Bitwise
  BinaryVecVecBitwiseAnd = OP_BINARY_VEC_VEC_BITWISE_AND,
  BinaryVecVecBitwiseOr = OP_BINARY_VEC_VEC_BITWISE_OR,
  BinaryVecVecBitwiseXor = OP_BINARY_VEC_VEC_BITWISE_XOR,
  BinaryVecVecLeftShift = OP_BINARY_VEC_VEC_LEFT_SHIFT,
  BinaryVecVecRightShift = OP_BINARY_VEC_VEC_RIGHT_SHIFT,

  // Logical
  BinaryVecVecLogicalAnd = OP_BINARY_VEC_VEC_LOGICAL_AND,
  BinaryVecVecLogicalOr = OP_BINARY_VEC_VEC_LOGICAL_OR,
  BinaryVecVecLogicalXor = OP_BINARY_VEC_VEC_LOGICAL_XOR,

  // Math
  BinaryVecVecAtan2 = OP_BINARY_VEC_VEC_ATAN2,
  BinaryVecVecHypot = OP_BINARY_VEC_VEC_HYPOT,
  BinaryVecVecCopysign = OP_BINARY_VEC_VEC_COPYSIGN,
  BinaryVecVecFmod = OP_BINARY_VEC_VEC_FMOD,
  BinaryVecVecLogaddexp = OP_BINARY_VEC_VEC_LOGADDEXP,
  BinaryVecVecLogaddexp2 = OP_BINARY_VEC_VEC_LOGADDEXP2,

  // ===========================================================================
  // Binary vec-scalar operations (30-59)
  // ===========================================================================

  // Arithmetic
  BinaryVecScalarAdd = OP_BINARY_VEC_SCALAR_ADD,
  BinaryVecScalarSub = OP_BINARY_VEC_SCALAR_SUB,
  BinaryVecScalarMul = OP_BINARY_VEC_SCALAR_MUL,
  BinaryVecScalarDiv = OP_BINARY_VEC_SCALAR_DIV,
  BinaryVecScalarMod = OP_BINARY_VEC_SCALAR_MOD,
  BinaryVecScalarPow = OP_BINARY_VEC_SCALAR_POW,
  BinaryVecScalarFloorDiv = OP_BINARY_VEC_SCALAR_FLOOR_DIV,

  // Comparison
  BinaryVecScalarEqual = OP_BINARY_VEC_SCALAR_EQUAL,
  BinaryVecScalarNotEqual = OP_BINARY_VEC_SCALAR_NOT_EQUAL,
  BinaryVecScalarLess = OP_BINARY_VEC_SCALAR_LESS,
  BinaryVecScalarLessEqual = OP_BINARY_VEC_SCALAR_LESS_EQUAL,
  BinaryVecScalarGreater = OP_BINARY_VEC_SCALAR_GREATER,
  BinaryVecScalarGreaterEqual = OP_BINARY_VEC_SCALAR_GREATER_EQUAL,

  // Min/Max
  BinaryVecScalarMin = OP_BINARY_VEC_SCALAR_MIN,
  BinaryVecScalarMax = OP_BINARY_VEC_SCALAR_MAX,

  // Bitwise
  BinaryVecScalarBitwiseAnd = OP_BINARY_VEC_SCALAR_BITWISE_AND,
  BinaryVecScalarBitwiseOr = OP_BINARY_VEC_SCALAR_BITWISE_OR,
  BinaryVecScalarBitwiseXor = OP_BINARY_VEC_SCALAR_BITWISE_XOR,
  BinaryVecScalarLeftShift = OP_BINARY_VEC_SCALAR_LEFT_SHIFT,
  BinaryVecScalarRightShift = OP_BINARY_VEC_SCALAR_RIGHT_SHIFT,

  // Logical
  BinaryVecScalarLogicalAnd = OP_BINARY_VEC_SCALAR_LOGICAL_AND,
  BinaryVecScalarLogicalOr = OP_BINARY_VEC_SCALAR_LOGICAL_OR,
  BinaryVecScalarLogicalXor = OP_BINARY_VEC_SCALAR_LOGICAL_XOR,

  // Math
  BinaryVecScalarAtan2 = OP_BINARY_VEC_SCALAR_ATAN2,
  BinaryVecScalarHypot = OP_BINARY_VEC_SCALAR_HYPOT,
  BinaryVecScalarCopysign = OP_BINARY_VEC_SCALAR_COPYSIGN,
  BinaryVecScalarFmod = OP_BINARY_VEC_SCALAR_FMOD,

  // Activation
  BinaryVecScalarLeakyRelu = OP_BINARY_VEC_SCALAR_LEAKY_RELU,
  BinaryVecScalarPrelu = OP_BINARY_VEC_SCALAR_PRELU,
  BinaryVecScalarHardshrink = OP_BINARY_VEC_SCALAR_HARDSHRINK,

  // ===========================================================================
  // Unary operations (60-96)
  // ===========================================================================

  // Basic
  UnaryNeg = OP_UNARY_NEG,
  UnaryAbs = OP_UNARY_ABS,
  UnarySqrt = OP_UNARY_SQRT,
  UnarySquare = OP_UNARY_SQUARE,
  UnaryReciprocal = OP_UNARY_RECIPROCAL,
  UnarySign = OP_UNARY_SIGN,

  // Exponential/Logarithmic
  UnaryExp = OP_UNARY_EXP,
  UnaryExp2 = OP_UNARY_EXP2,
  UnaryExpm1 = OP_UNARY_EXPM1,
  UnaryLog = OP_UNARY_LOG,
  UnaryLog2 = OP_UNARY_LOG2,
  UnaryLog10 = OP_UNARY_LOG10,
  UnaryLog1p = OP_UNARY_LOG1P,

  // Trigonometric
  UnarySin = OP_UNARY_SIN,
  UnaryCos = OP_UNARY_COS,
  UnaryTan = OP_UNARY_TAN,
  UnaryAsin = OP_UNARY_ASIN,
  UnaryAcos = OP_UNARY_ACOS,
  UnaryAtan = OP_UNARY_ATAN,

  // Hyperbolic
  UnarySinh = OP_UNARY_SINH,
  UnaryCosh = OP_UNARY_COSH,
  UnaryTanh = OP_UNARY_TANH,

  // Rounding
  UnaryFloor = OP_UNARY_FLOOR,
  UnaryCeil = OP_UNARY_CEIL,
  UnaryRound = OP_UNARY_ROUND,

  // Special Math
  UnaryCbrt = OP_UNARY_CBRT,
  UnaryDegrees = OP_UNARY_DEGREES,
  UnaryRadians = OP_UNARY_RADIANS,

  // Logical/Bitwise
  UnaryLogicalNot = OP_UNARY_LOGICAL_NOT,
  UnaryBitwiseNot = OP_UNARY_BITWISE_NOT,

  // Activation Functions
  UnaryRelu = OP_UNARY_RELU,
  UnarySigmoid = OP_UNARY_SIGMOID,
  UnaryGelu = OP_UNARY_GELU,
  UnarySilu = OP_UNARY_SILU,
  UnarySoftplus = OP_UNARY_SOFTPLUS,

  // Check Operations
  UnaryIsNan = OP_UNARY_ISNAN,
  UnaryIsInf = OP_UNARY_ISINF,

  // ===========================================================================
  // Extended unary operations (160-199)
  // ===========================================================================

  // Extended Activations
  UnaryRelu6 = OP_UNARY_RELU6,
  UnaryElu = OP_UNARY_ELU,
  UnarySelu = OP_UNARY_SELU,
  UnaryCelu = OP_UNARY_CELU,
  UnaryMish = OP_UNARY_MISH,
  UnaryHardswish = OP_UNARY_HARDSWISH,
  UnaryHardsigmoid = OP_UNARY_HARDSIGMOID,
  UnaryHardtanh = OP_UNARY_HARDTANH,
  UnarySoftsign = OP_UNARY_SOFTSIGN,
  UnaryLogSigmoid = OP_UNARY_LOGSIGMOID,
  UnaryTanhshrink = OP_UNARY_TANHSHRINK,

  // Extended Math
  UnaryRsqrt = OP_UNARY_RSQRT,
  UnaryTrunc = OP_UNARY_TRUNC,
  UnaryFrac = OP_UNARY_FRAC,
  UnaryAsinh = OP_UNARY_ASINH,
  UnaryAcosh = OP_UNARY_ACOSH,
  UnaryAtanh = OP_UNARY_ATANH,
  UnaryIsFinite = OP_UNARY_ISFINITE,

  // ===========================================================================
  // Extended binary vec-scalar operations (200-219)
  // ===========================================================================

  BinaryVecScalarSoftshrink = OP_BINARY_VEC_SCALAR_SOFTSHRINK,
  BinaryVecScalarLogaddexp = OP_BINARY_VEC_SCALAR_LOGADDEXP,
  BinaryVecScalarLogaddexp2 = OP_BINARY_VEC_SCALAR_LOGADDEXP2,

  // ===========================================================================
  // Ternary operations (100-109)
  // ===========================================================================

  TernaryClamp = OP_TERNARY_CLAMP,
  TernarySelect = OP_TERNARY_SELECT,

  // ===========================================================================
  // Reduction operations (110-119)
  // ===========================================================================

  ReduceSum = OP_REDUCE_SUM,
  ReduceMean = OP_REDUCE_MEAN,
  ReduceMin = OP_REDUCE_MIN,
  ReduceMax = OP_REDUCE_MAX,
  ReduceProd = OP_REDUCE_PROD,
  ReduceAny = OP_REDUCE_ANY,
  ReduceAll = OP_REDUCE_ALL,
  ReduceDimSum = OP_REDUCE_DIM_SUM,
  ReduceDimMean = OP_REDUCE_DIM_MEAN,
  ReduceDimMin = OP_REDUCE_DIM_MIN,
  ReduceDimMax = OP_REDUCE_DIM_MAX,
  ReduceDimProd = OP_REDUCE_DIM_PROD,
  ReduceDimAny = OP_REDUCE_DIM_ANY,
  ReduceDimAll = OP_REDUCE_DIM_ALL,

  // Argmax/Argmin
  ReduceArgmax = OP_REDUCE_ARGMAX,
  ReduceArgmin = OP_REDUCE_ARGMIN,
  ReduceDimArgmax = OP_REDUCE_DIM_ARGMAX,
  ReduceDimArgmin = OP_REDUCE_DIM_ARGMIN,

  // ===========================================================================
  // Cumulative/scan operations (240-259)
  // ===========================================================================

  CumSum = OP_CUMSUM,
  CumProd = OP_CUMPROD,

  // ===========================================================================
  // Matrix operations (120-122)
  // ===========================================================================

  MatMul = OP_MATMUL,
  Transpose = OP_TRANSPOSE,
  Dot = OP_DOT,

  // ===========================================================================
  // Tensor manipulation operations (130-139)
  // ===========================================================================

  Concat = OP_CONCAT,
  Stack = OP_STACK,
  Flatten = OP_FLATTEN,

  // ===========================================================================
  // Norm operations (140-149)
  // ===========================================================================

  Norm = OP_NORM,
  NormDim = OP_NORM_DIM,

  // ===========================================================================
  // Tensor creation operations (150-159)
  // ===========================================================================

  Arange = OP_ARANGE,
  Linspace = OP_LINSPACE,
  Zeros = OP_ZEROS,
  Ones = OP_ONES,
  Full = OP_FULL,
};

/**
 * Returns a string name for the given operator.
 * @param op The operator enum.
 * @return A human-readable name for the operator.
 */
const char *operatorName(OperatorEnum op);

} // namespace cut
