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
  // Binary operations (0-32) - unified for all binary variants
  // ===========================================================================

  // Arithmetic
  BinaryAdd = OP_BINARY_ADD,
  BinarySub = OP_BINARY_SUB,
  BinaryMul = OP_BINARY_MUL,
  BinaryDiv = OP_BINARY_DIV,
  BinaryMod = OP_BINARY_MOD,
  BinaryPow = OP_BINARY_POW,
  BinaryFloorDiv = OP_BINARY_FLOOR_DIV,

  // Comparison
  BinaryEqual = OP_BINARY_EQUAL,
  BinaryNotEqual = OP_BINARY_NOT_EQUAL,
  BinaryLess = OP_BINARY_LESS,
  BinaryLessEqual = OP_BINARY_LESS_EQUAL,
  BinaryGreater = OP_BINARY_GREATER,
  BinaryGreaterEqual = OP_BINARY_GREATER_EQUAL,

  // Min/Max
  BinaryMin = OP_BINARY_MIN,
  BinaryMax = OP_BINARY_MAX,

  // Bitwise
  BinaryBitwiseAnd = OP_BINARY_BITWISE_AND,
  BinaryBitwiseOr = OP_BINARY_BITWISE_OR,
  BinaryBitwiseXor = OP_BINARY_BITWISE_XOR,
  BinaryLeftShift = OP_BINARY_LEFT_SHIFT,
  BinaryRightShift = OP_BINARY_RIGHT_SHIFT,

  // Logical
  BinaryLogicalAnd = OP_BINARY_LOGICAL_AND,
  BinaryLogicalOr = OP_BINARY_LOGICAL_OR,
  BinaryLogicalXor = OP_BINARY_LOGICAL_XOR,

  // Math
  BinaryAtan2 = OP_BINARY_ATAN2,
  BinaryHypot = OP_BINARY_HYPOT,
  BinaryCopysign = OP_BINARY_COPYSIGN,
  BinaryFmod = OP_BINARY_FMOD,

  // Activation
  BinaryLeakyRelu = OP_BINARY_LEAKY_RELU,
  BinaryPrelu = OP_BINARY_PRELU,
  BinaryHardshrink = OP_BINARY_HARDSHRINK,
  BinarySoftshrink = OP_BINARY_SOFTSHRINK,
  BinaryLogaddexp = OP_BINARY_LOGADDEXP,
  BinaryLogaddexp2 = OP_BINARY_LOGADDEXP2,

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
  // Matmul variants are selected by index (see matmul_variants.json).
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
  Cast = OP_CAST,

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
  // Normalization operations (310-314)
  // ===========================================================================

  LayerNorm = OP_LAYER_NORM,
  BatchNorm = OP_BATCH_NORM,
  RMSNorm = OP_RMS_NORM,
  ExtendedRMSNorm = OP_EXTENDED_RMS_NORM,
  MatMulSiLU = OP_MATMUL_SILU,

  // ===========================================================================
  // Embedding operations (320)
  // ===========================================================================

  Embedding = OP_EMBEDDING,

  // ===========================================================================
  // Padding operations (330)
  // ===========================================================================

  Pad = OP_PAD,

  // ===========================================================================
  // Expand operations (340)
  // ===========================================================================

  Expand = OP_EXPAND,

  // ===========================================================================
  // RoPE operations (350)
  // ===========================================================================

  RoPE = OP_ROPE,

  // ===========================================================================
  // Attention operations (360-361)
  // ===========================================================================

  CacheWrite = OP_CACHE_WRITE,
  Attention = OP_ATTENTION,

  // ===========================================================================
  // Quantized matmul operations (371)
  // ===========================================================================

  MatMulQ8 = OP_MATMUL_Q8,

  // ===========================================================================
  // Fused binary operations (380)
  // ===========================================================================

  FusedBinary = OP_FUSED_BINARY,
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
  int recommendedVariant =
      -1; ///< Recommended variant index (-1 = not applicable).
};

} // namespace cut
