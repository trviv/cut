#pragma once

#include <cstdint>

// Include shared definitions (compatible with GLSL)
#include "ComputeOpsShared.h"

namespace cut {

/**
 * Scalar data type for operations.
 * Values match DTYPE_* defines in ComputeOps.glsl for shader compatibility.
 */
enum ScalarDataType {
  Float = DTYPE_FLOAT,
  Half = DTYPE_HALF,
  UInt = DTYPE_UINT,
  Int = DTYPE_INT,
};

/**
 * Operator enum for built-in compute operations.
 * Used by both GPU shaders and CPU kernels.
 * Values match OP_* defines in ComputeOps.glsl for shader compatibility.
 */
enum OperatorEnum {
  // Binary arithmetic operations (vec-vec)
  BinaryVecVecAdd = OP_BINARY_VEC_VEC_ADD,
  BinaryVecVecSub = OP_BINARY_VEC_VEC_SUB,
  BinaryVecVecMul = OP_BINARY_VEC_VEC_MUL,
  BinaryVecVecDiv = OP_BINARY_VEC_VEC_DIV,
  BinaryVecVecMod = OP_BINARY_VEC_VEC_MOD,
  BinaryVecVecPow = OP_BINARY_VEC_VEC_POW,
  BinaryVecVecFloorDiv = OP_BINARY_VEC_VEC_FLOOR_DIV,

  // Binary comparison operations (vec-vec)
  BinaryVecVecEqual = OP_BINARY_VEC_VEC_EQUAL,
  BinaryVecVecNotEqual = OP_BINARY_VEC_VEC_NOT_EQUAL,
  BinaryVecVecLess = OP_BINARY_VEC_VEC_LESS,
  BinaryVecVecLessEqual = OP_BINARY_VEC_VEC_LESS_EQUAL,
  BinaryVecVecGreater = OP_BINARY_VEC_VEC_GREATER,
  BinaryVecVecGreaterEqual = OP_BINARY_VEC_VEC_GREATER_EQUAL,

  // Binary min/max operations (vec-vec)
  BinaryVecVecMin = OP_BINARY_VEC_VEC_MIN,
  BinaryVecVecMax = OP_BINARY_VEC_VEC_MAX,

  // Binary arithmetic operations (vec-scalar)
  BinaryVecScalarAdd = OP_BINARY_VEC_SCALAR_ADD,
  BinaryVecScalarSub = OP_BINARY_VEC_SCALAR_SUB,
  BinaryVecScalarMul = OP_BINARY_VEC_SCALAR_MUL,
  BinaryVecScalarDiv = OP_BINARY_VEC_SCALAR_DIV,
  BinaryVecScalarMod = OP_BINARY_VEC_SCALAR_MOD,
  BinaryVecScalarPow = OP_BINARY_VEC_SCALAR_POW,
  BinaryVecScalarFloorDiv = OP_BINARY_VEC_SCALAR_FLOOR_DIV,

  // Binary comparison operations (vec-scalar)
  BinaryVecScalarEqual = OP_BINARY_VEC_SCALAR_EQUAL,
  BinaryVecScalarNotEqual = OP_BINARY_VEC_SCALAR_NOT_EQUAL,
  BinaryVecScalarLess = OP_BINARY_VEC_SCALAR_LESS,
  BinaryVecScalarLessEqual = OP_BINARY_VEC_SCALAR_LESS_EQUAL,
  BinaryVecScalarGreater = OP_BINARY_VEC_SCALAR_GREATER,
  BinaryVecScalarGreaterEqual = OP_BINARY_VEC_SCALAR_GREATER_EQUAL,

  // Binary min/max operations (vec-scalar)
  BinaryVecScalarMin = OP_BINARY_VEC_SCALAR_MIN,
  BinaryVecScalarMax = OP_BINARY_VEC_SCALAR_MAX,

  // Unary operations
  UnaryNeg = OP_UNARY_NEG,
  UnaryAbs = OP_UNARY_ABS,
  UnarySqrt = OP_UNARY_SQRT,
  UnaryExp = OP_UNARY_EXP,
  UnaryLog = OP_UNARY_LOG,
  UnaryLog2 = OP_UNARY_LOG2,
  UnaryLog10 = OP_UNARY_LOG10,
  UnarySin = OP_UNARY_SIN,
  UnaryCos = OP_UNARY_COS,
  UnaryTan = OP_UNARY_TAN,
  UnaryAsin = OP_UNARY_ASIN,
  UnaryAcos = OP_UNARY_ACOS,
  UnaryAtan = OP_UNARY_ATAN,
  UnarySinh = OP_UNARY_SINH,
  UnaryCosh = OP_UNARY_COSH,
  UnaryTanh = OP_UNARY_TANH,
  UnaryFloor = OP_UNARY_FLOOR,
  UnaryCeil = OP_UNARY_CEIL,
  UnaryRound = OP_UNARY_ROUND,
  UnarySign = OP_UNARY_SIGN,
  UnaryReciprocal = OP_UNARY_RECIPROCAL,
  UnarySquare = OP_UNARY_SQUARE,
};

} // namespace cut
