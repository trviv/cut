
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

} // namespace cut
