#pragma once

#include <cstdint>
#include <vector>

namespace cut {

/// A specialization-constant value extracted from SPIR-V.
struct CudaSpecValue {
  uint32_t id;    ///< SpecId decoration value.
  uint32_t value; ///< Current literal (already patched by the operator).
};

/// Computes a stable hash that identifies a kernel independent of its
/// specialization-constant values, and extracts those values.
///
/// All OpSpecConstant literals whose result id carries a SpecId decoration are
/// zeroed before hashing (FNV-1a/64 over the little-endian word bytes), so the
/// patched runtime SPIR-V and the unpatched build-time .spv hash identically.
/// The pre-normalization literals are returned in @p outSpecs (ascending id).
///
/// The matching build-time implementation lives in scripts/embed_cuda_kernels.py
/// — the two MUST stay in lockstep.
uint64_t cudaNormalizedSpirvHash(const std::vector<uint32_t> &spirv,
                                 std::vector<CudaSpecValue> &outSpecs);

} // namespace cut
