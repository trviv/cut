
#pragma once

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cut {

enum ShaderEnum {
  VECTOR_ADD,

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

/*
 * Function returns spirv encoding for an in-build shader.
 */
std::vector<uint32_t> getShader(const ShaderEnum shader);

/*
 * Function returns spirv encoding for a runtime-generated shader.
 * Returns std::nullopt if the shader enum is not handled by ShadersGenerated.
 */
std::optional<std::vector<uint32_t>>
getGeneratedShader(const ShaderEnum shader);

/*
 * Function returns spirv encoding for a pre-compiled shader.
 * Returns std::nullopt if the shader enum is not handled by CompiledShaders.
 */
std::optional<std::vector<uint32_t>> getCompiledShader(const ShaderEnum shader);

} // namespace cut