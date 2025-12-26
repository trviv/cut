#pragma once

#include <cstdint>

namespace cut {

/**
 * Scalar data type for operations.
 */
enum ScalarDataType {
  Float,
  Half,
  UInt,
  Int,
};

/**
 * Operator enum for built-in compute operations.
 * Used by both GPU shaders and CPU kernels.
 */
enum OperatorEnum {
  // Binary arithmetic operations (vec-vec)
  BinaryVecVecAdd,
  BinaryVecVecSub,
  BinaryVecVecMul,
  BinaryVecVecDiv,
  BinaryVecVecMod,
  BinaryVecVecPow,
  BinaryVecVecFloorDiv,

  // Binary comparison operations (vec-vec)
  BinaryVecVecEqual,
  BinaryVecVecNotEqual,
  BinaryVecVecLess,
  BinaryVecVecLessEqual,
  BinaryVecVecGreater,
  BinaryVecVecGreaterEqual,

  // Binary min/max operations (vec-vec)
  BinaryVecVecMin,
  BinaryVecVecMax,

  // Binary arithmetic operations (vec-scalar)
  BinaryVecScalarAdd,
  BinaryVecScalarSub,
  BinaryVecScalarMul,
  BinaryVecScalarDiv,
  BinaryVecScalarMod,
  BinaryVecScalarPow,
  BinaryVecScalarFloorDiv,

  // Binary comparison operations (vec-scalar)
  BinaryVecScalarEqual,
  BinaryVecScalarNotEqual,
  BinaryVecScalarLess,
  BinaryVecScalarLessEqual,
  BinaryVecScalarGreater,
  BinaryVecScalarGreaterEqual,

  // Binary min/max operations (vec-scalar)
  BinaryVecScalarMin,
  BinaryVecScalarMax,

  // Unary operations
  UnaryNeg,
  UnaryAbs,
  UnarySqrt,
  UnaryExp,
  UnaryLog,
  UnaryLog2,
  UnaryLog10,
  UnarySin,
  UnaryCos,
  UnaryTan,
  UnaryAsin,
  UnaryAcos,
  UnaryAtan,
  UnarySinh,
  UnaryCosh,
  UnaryTanh,
  UnaryFloor,
  UnaryCeil,
  UnaryRound,
  UnarySign,
  UnaryReciprocal,
  UnarySquare,
};

} // namespace cut
