
#pragma once

#include <ComputeCommon.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace cut {

std::optional<std::vector<uint32_t>>
compiledMatMul(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulNaive(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulRegTiled(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulTiled2x2(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulT8R2x2(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulT8R4x4(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulT16R4x4(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulT16R8x8(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulT32R2x2(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulSimdR4x4(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulSimdR4x8(const DataType datatype = DataType::Float32);

std::optional<std::vector<uint32_t>>
compiledMatMulSimdR8x8(const DataType datatype = DataType::Float32);

} // namespace cut
