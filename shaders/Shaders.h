
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
 * Pre-compiled shader functions.
 * Each function returns spirv encoding for a specific compiled shader.
 * The datatype parameter selects the SPIR-V variant for the requested type.
 * Returns std::nullopt if the datatype is not supported.
 */
std::optional<std::vector<uint32_t>>
compiledBinaryVecVec(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledBinaryVecScalar(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledUnary(const DataType datatype = DataType::Float32);

/*
 * Validates execution sizes for an operator and returns the resolved size.
 * For elementwise operators (unary, binary, ternary), all buffer execution
 * sizes must match. Future operators (e.g., matmul, reduce) may have
 * different validation rules based on their semantics.
 *
 * @param op The operator being executed.
 * @param execSizes The execution sizes of all buffer bindings.
 * @return The validated execution size.
 * @throws std::runtime_error if sizes are invalid for the operator.
 */
size_t validateExecutionSize(OperatorEnum op,
                             const std::vector<size_t> &execSizes);

} // namespace cut
