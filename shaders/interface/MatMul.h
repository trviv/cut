
#pragma once

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace cut {

/// Returns the compiled SPIR-V for the given matmul operator variant.
std::optional<std::vector<uint32_t>>
getCompiledMatMul(OperatorEnum op, DataType datatype = DataType::Float32);

} // namespace cut
