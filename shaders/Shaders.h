
#pragma once

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cut {

// Backward compatibility alias
using ShaderEnum = OperatorEnum;

/*
 * Function returns spirv encoding for an in-build shader.
 */
std::vector<uint32_t> getShader(const OperatorEnum shader,
                                const DataType datatype = DataType::Float32);

/*
 * Function returns spirv encoding for a runtime-generated shader.
 * Returns std::nullopt if the shader enum is not handled by ShadersGenerated.
 */
std::optional<std::vector<uint32_t>>
getGeneratedShader(const OperatorEnum shader,
                   const DataType datatype = DataType::Float32);

/*
 * Function returns spirv encoding for a pre-compiled shader.
 * Returns std::nullopt if the shader enum is not handled by CompiledShaders.
 */
std::optional<std::vector<uint32_t>>
getCompiledShader(const OperatorEnum shader);

/*
 * Returns the scaled dispatch size for a shader.
 * When shaders use vectorized types (vec4), the dispatch size is scaled down
 * by 4 since each invocation processes 4 elements.
 */
uint32_t getScaledDispatchSize(uint32_t dispatchSize,
                               const OperatorEnum shader,
                               const DataType datatype = DataType::Float32);

} // namespace cut
