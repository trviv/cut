
#include <Shaders.h>

namespace cut {

// Forward declarations for compiled shader functions (generated in
// CompiledShaders.cpp).
std::optional<std::vector<uint32_t>> compiledMatMul(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulNaive(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulRegTiled(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulTiled2x2(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulT8R2x2(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulT8R4x4(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulT16R4x4(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulT16R8x8(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulT32R2x2(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulSimdR4x4(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulSimdR4x8(DataType datatype);
std::optional<std::vector<uint32_t>> compiledMatMulSimdR8x8(DataType datatype);

bool isMatMulOp(OperatorEnum op) {
  return op == MatMul || op == MatMulNaive || op == MatMulRegTiled ||
         op == MatMulTiled2x2 || op == MatMulT8R2x2 || op == MatMulT8R4x4 ||
         op == MatMulT16R4x4 || op == MatMulT16R8x8 || op == MatMulT32R2x2 ||
         op == MatMulSimdR4x4 || op == MatMulSimdR4x8 || op == MatMulSimdR8x8;
}

std::optional<std::vector<uint32_t>> getCompiledMatMul(OperatorEnum op,
                                                       DataType datatype) {
  switch (op) {
  case MatMul:
    return compiledMatMul(datatype);
  case MatMulNaive:
    return compiledMatMulNaive(datatype);
  case MatMulRegTiled:
    return compiledMatMulRegTiled(datatype);
  case MatMulTiled2x2:
    return compiledMatMulTiled2x2(datatype);
  case MatMulT8R2x2:
    return compiledMatMulT8R2x2(datatype);
  case MatMulT8R4x4:
    return compiledMatMulT8R4x4(datatype);
  case MatMulT16R4x4:
    return compiledMatMulT16R4x4(datatype);
  case MatMulT16R8x8:
    return compiledMatMulT16R8x8(datatype);
  case MatMulT32R2x2:
    return compiledMatMulT32R2x2(datatype);
  case MatMulSimdR4x4:
    return compiledMatMulSimdR4x4(datatype);
  case MatMulSimdR4x8:
    return compiledMatMulSimdR4x8(datatype);
  case MatMulSimdR8x8:
    return compiledMatMulSimdR8x8(datatype);
  default:
    return std::nullopt;
  }
}

} // namespace cut
