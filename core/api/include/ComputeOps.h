#pragma once

#include <cstddef>
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
  // Binary vec-scalar operations (29-61)
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
  // Unary operations (100-154)
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
  // Extended unary operations (137-154)
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
  // Extended binary vec-scalar operations (59-61)
  // ===========================================================================

  BinaryVecScalarSoftshrink = OP_BINARY_VEC_SCALAR_SOFTSHRINK,
  BinaryVecScalarLogaddexp = OP_BINARY_VEC_SCALAR_LOGADDEXP,
  BinaryVecScalarLogaddexp2 = OP_BINARY_VEC_SCALAR_LOGADDEXP2,

  // ===========================================================================
  // Ternary operations (62-63)
  // ===========================================================================

  TernaryClamp = OP_TERNARY_CLAMP,
  TernarySelect = OP_TERNARY_SELECT,

  // ===========================================================================
  // Reduction operations (200-217)
  // ===========================================================================

  ReduceSum = OP_REDUCE_SUM,
  ReduceMean = OP_REDUCE_MEAN,
  ReduceMin = OP_REDUCE_MIN,
  ReduceMax = OP_REDUCE_MAX,
  ReduceProd = OP_REDUCE_PROD,
  ReduceAny = OP_REDUCE_ANY,
  ReduceAll = OP_REDUCE_ALL,

  // Argmax/Argmin
  ReduceArgmax = OP_REDUCE_ARGMAX,
  ReduceArgmin = OP_REDUCE_ARGMIN,

  // ===========================================================================
  // Cumulative/scan operations (240-259)
  // ===========================================================================

  CumSum = OP_CUMSUM,
  CumProd = OP_CUMPROD,

  // ===========================================================================
  // Prefix scan operations (260-269)
  // ===========================================================================

  PrefixScanExclusiveSum = OP_PREFIX_SCAN_EXCLUSIVE_SUM,
  PrefixScanInclusiveSum = OP_PREFIX_SCAN_INCLUSIVE_SUM,

  // ===========================================================================
  // Sort operations (270-279)
  // ===========================================================================

  SortBitonic = OP_SORT_BITONIC,
  SortRadix = OP_SORT_RADIX,

  // ===========================================================================
  // Matrix operations (64-66)
  // ===========================================================================

  MatMul = OP_MATMUL,
  MatMulNaive = OP_MATMUL_NAIVE,
  MatMulRegTiled = OP_MATMUL_REG_TILED,
  MatMulTiled2x2 = OP_MATMUL_TILED_2X2,
  MatMulT8R2x2 = OP_MATMUL_T8_R2X2,
  MatMulT8R4x4 = OP_MATMUL_T8_R4X4,
  MatMulT16R4x4 = OP_MATMUL_T16_R4X4,
  MatMulT16R8x8 = OP_MATMUL_T16_R8X8,
  MatMulT32R2x2 = OP_MATMUL_T32_R2X2,
  MatMulSimdR4x4 = OP_MATMUL_SIMD_R4X4,
  MatMulSimdR4x8 = OP_MATMUL_SIMD_R4X8,
  MatMulSimdR8x8 = OP_MATMUL_SIMD_R8X8,
  Transpose = OP_TRANSPOSE,
  Dot = OP_DOT,

  // ===========================================================================
  // Tensor manipulation operations (67-69)
  // ===========================================================================

  Concat = OP_CONCAT,
  Stack = OP_STACK,
  Flatten = OP_FLATTEN,

  // ===========================================================================
  // Norm operations (70-71)
  // ===========================================================================

  Norm = OP_NORM,
  NormDim = OP_NORM_DIM,

  // ===========================================================================
  // Tensor creation operations (72-77)
  // ===========================================================================

  Arange = OP_ARANGE,
  Linspace = OP_LINSPACE,
  Zeros = OP_ZEROS,
  Ones = OP_ONES,
  Full = OP_FULL,
  Copy = OP_COPY,

  // ===========================================================================
  // Dispatcher internal shader templates (280-299)
  // ===========================================================================

  // Multi-workgroup reduce (parameterized templates)
  InternalPartialReduce = OP_INTERNAL_PARTIAL_REDUCE,
  InternalFinalReduce = OP_INTERNAL_FINAL_REDUCE,

  // Prefix scan (three-pass)
  InternalScanPerWg = OP_INTERNAL_SCAN_PER_WG,
  InternalScanPartialSums = OP_INTERNAL_SCAN_PARTIAL_SUMS,
  InternalScanPropagate = OP_INTERNAL_SCAN_PROPAGATE,

  // Bitonic sort
  InternalBitonicStep = OP_INTERNAL_BITONIC_STEP,
  InternalBitonicPadInit = OP_INTERNAL_BITONIC_PAD_INIT,
  InternalBitonicCopyBack = OP_INTERNAL_BITONIC_COPY_BACK,

  // Radix sort
  InternalRadixHistogram = OP_INTERNAL_RADIX_HISTOGRAM,
  InternalRadixScatter = OP_INTERNAL_RADIX_SCATTER,

  // Utility
  InternalFillUint = OP_INTERNAL_FILL_UINT,
  InternalScanUint = OP_INTERNAL_SCAN_UINT,

  // ===========================================================================
  // Convolution operations (300-301)
  // ===========================================================================

  Conv1D = OP_CONV1D,
  Conv2D = OP_CONV2D,

  // ===========================================================================
  // Pooling operations (302-303)
  // ===========================================================================

  MaxPool2D = OP_MAX_POOL2D,
  AvgPool2D = OP_AVG_POOL2D,

  // ===========================================================================
  // Normalization operations (310-311)
  // ===========================================================================

  LayerNorm = OP_LAYER_NORM,
  BatchNorm = OP_BATCH_NORM,

  // ===========================================================================
  // Embedding operations (320)
  // ===========================================================================

  Embedding = OP_EMBEDDING,

  // ===========================================================================
  // Padding operations (330)
  // ===========================================================================

  Pad = OP_PAD,
};

/**
 * Returns a string name for the given operator.
 * @param op The operator enum.
 * @return A human-readable name for the operator.
 */
const char *operatorName(OperatorEnum op);

/**
 * Result of execution size validation, containing the resolved size and the
 * recommended operator to use (which may differ from the requested operator
 * based on input shapes).
 */
struct ExecutionConfig {
  size_t size;     ///< Validated execution size.
  OperatorEnum op; ///< Recommended operator for the given shapes.
};

} // namespace cut
