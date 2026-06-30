#pragma once

#include <cstddef>
#include <cstdint>

namespace cut {

/// One transpiled CUDA kernel, keyed by the normalized hash of its source
/// SPIR-V (spec-constant literals zeroed). Generated into CompiledCudaKernels.cpp.
struct CudaKernelEntry {
  uint64_t hash;       ///< Normalized SPIR-V hash (see cudaNormalizedSpirvHash).
  const char *source;  ///< CUDA C++ kernel source (transpiled HLSL).
  const char *entry;   ///< extern "C" kernel entry-point symbol name.
};

/// Looks up a transpiled kernel by normalized SPIR-V hash; null if absent.
const CudaKernelEntry *lookupCudaKernelByHash(uint64_t hash);

/// Number of registered CUDA kernels.
size_t cudaKernelCount();

/// The shared CUDA prelude source (vector-type operators + HLSL intrinsics).
const char *cudaPreludeSource();

/// The shared operator-enum header source (ComputeOpsShared.h contents).
const char *cudaEnumsSource();

} // namespace cut
