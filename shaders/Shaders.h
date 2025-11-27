
#pragma once

#include <fstream>
#include <string>
#include <unordered_map>

namespace cut {

enum ShaderEnum {
  VECTOR_ADD,
};

/*
 * Function returns spirv encoding for an in-build shader.
 */
static std::vector<uint32_t> getShader(const ShaderEnum shader);

} // namespace cut
