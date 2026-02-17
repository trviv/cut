
#pragma once

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <cstdint>
#include <optional>
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

std::optional<std::vector<uint32_t>>
compiledMatMul(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledTranspose(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledReduce(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledReduceDim(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledReduceArg(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledReduceDimArg(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledCumOp(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledDot(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledTernaryClamp(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledTernarySelect(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledNorm(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledArange(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledFill(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledCopy(const DataType datatype = DataType::Float32);

// Dispatcher internal compiled shaders (prefix scan, bitonic sort, radix sort,
// utility)
std::optional<std::vector<uint32_t>>
compiledPartialReduce(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledFinalReduce(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledScanPerWg(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledScanPartialSums(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledScanPropagate(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledBitonicStep(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledBitonicPadInit(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledBitonicCopyBack(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledRadixHistogram(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledRadixScatter(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledFillUint(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledScanUint(const DataType datatype = DataType::Float32);

// Convolution compiled shaders
std::optional<std::vector<uint32_t>>
compiledConv1D(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledConv2D(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledConvTranspose2D(const DataType datatype = DataType::Float32);

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
