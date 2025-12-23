
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
  BinaryVecVecAdd,
  BinaryVecVecSub,
  BinaryVecVecMul,
  BinaryVecVecDiv,
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

} // namespace cut