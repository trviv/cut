
#pragma once

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <cstdint>
#include <optional>
#include <vector>

// Variant dispatch tables and lookup functions
#include "impl/avgpool2d/AvgPool2DVariants.generated.h"
#include "impl/conv1d/Conv1DVariants.generated.h"
#include "impl/conv2d/Conv2DVariants.generated.h"
#include "impl/matmul/MatMulQ4Variants.generated.h"
#include "impl/matmul/MatMulQ8Variants.generated.h"
#include "impl/matmul/MatMulVariants.generated.h"
#include "impl/maxpool2d/MaxPool2DVariants.generated.h"
#include "impl/reducedim/ReduceDimVariants.generated.h"
#include "impl/transpose/TransposeVariants.generated.h"

// Generated forward declarations for compiled shader functions
#include "impl/binary/BinaryShaders.generated.h"
#include "impl/creation/CreationShaders.generated.h"
#include "impl/memory/MemoryShaders.generated.h"
#include "impl/reduce/ReduceShaders.generated.h"
#include "impl/scan/ScanShaders.generated.h"
#include "impl/sort/SortShaders.generated.h"
#include "impl/ternary/TernaryShaders.generated.h"
#include "impl/unary/UnaryShaders.generated.h"

namespace cut {

// Backward compatibility alias
using ShaderEnum = OperatorEnum;

/*
 * Function returns spirv encoding for an in-build shader.
 */
std::vector<uint32_t> getShader(const OperatorEnum shader,
                                const DataType datatype = DataType::Float32);

/**
 * Returns SPIR-V for a dimension-wise reduction shader.
 * Compiles the ReduceDim (or ReduceDimArg) shader template and patches
 * the specialization constant with the given base reduce op enum.
 *
 * @param reduceOp Base reduce op (e.g. ReduceSum, ReduceArgmax).
 * @param datatype Data type variant.
 * @param variant Optional variant index for variant-based lookup.
 */
std::vector<uint32_t>
getDimReduceShader(const OperatorEnum reduceOp,
                   const DataType datatype = DataType::Float32,
                   std::optional<uint32_t> variant = {});

/**
 * Patch multiple specialization constants in a single SPIR-V scan.
 * Each pair is {specId, newValue}. More efficient than calling
 * patchSpecConstant repeatedly for fusion shaders with 2+ constants.
 */
void patchSpecConstants(
    std::vector<uint32_t> &spirv,
    const std::initializer_list<std::pair<uint32_t, uint32_t>> &patches);

/**
 * Patch a single specialization constant. Convenience wrapper around
 * patchSpecConstants for the common single-constant case.
 */
void patchSpecConstant(std::vector<uint32_t> &spirv,
                       uint32_t specId,
                       uint32_t newValue);

} // namespace cut
