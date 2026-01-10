/**
 * ShadersGenerated.cpp
 *
 * Main entry point for runtime shader generation in CUT.
 * This file coordinates shader generation across multiple specialized files:
 * - ShadersBasicOps.cpp: Binary and unary operations
 * - ShadersAdvancedOps.cpp: Reductions, matrix ops, conditional, tensor
 * creation
 *
 * All shader generation uses templates from ShaderUtils.h
 */

#include <ComputeCommon.h>
#include <Shaders.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace cut {

// Forward declarations from separate shader generation files
bool generateBasicOpShader(const OperatorEnum shader,
                           const DataType datatype,
                           std::string &shaderSource,
                           std::string &shaderName);

bool generateAdvancedOpShader(const OperatorEnum shader,
                              const DataType datatype,
                              std::string &shaderSource,
                              std::string &shaderName);

/// Enable/disable caching for generated SPIR-V shaders
constexpr bool kEnableShaderCache = true;

/// Cache for generated SPIR-V shaders, keyed by (OperatorEnum, DataType)
std::unordered_map<uint64_t, std::vector<uint32_t>> shaderCache;

// =============================================================================
// Main shader generation entry point
// =============================================================================

std::optional<std::vector<uint32_t>>
getGeneratedShader(const OperatorEnum shader, const DataType datatype) {
  // Check cache first
  if constexpr (kEnableShaderCache) {
    uint64_t cacheKey = makeCacheKey(shader, datatype);
    auto it = shaderCache.find(cacheKey);
    if (it != shaderCache.end()) {
      return it->second;
    }
  }

  std::string shaderSource;
  std::string shaderName = "generated_shader";

  // Try to generate shader from basic ops (binary vec-vec, vec-scalar, unary)
  if (generateBasicOpShader(shader, datatype, shaderSource, shaderName)) {
    // Shader generated successfully by ShadersBasicOps.cpp
    auto spirv =
        compileShaderToSpirv(shaderSource, shaderName, ShaderLanguage::GLSL);
    if constexpr (kEnableShaderCache) {
      shaderCache[makeCacheKey(shader, datatype)] = spirv;
    }
    return spirv;
  }

  // Try to generate shader from advanced ops (reductions, matrix, conditional,
  // etc.)
  if (generateAdvancedOpShader(shader, datatype, shaderSource, shaderName)) {
    // Shader generated successfully by ShadersAdvancedOps.cpp
    auto spirv =
        compileShaderToSpirv(shaderSource, shaderName, ShaderLanguage::GLSL);
    if constexpr (kEnableShaderCache) {
      shaderCache[makeCacheKey(shader, datatype)] = spirv;
    }
    return spirv;
  }

  // Shader not supported by runtime generation
  return std::nullopt;
}

} // namespace cut
