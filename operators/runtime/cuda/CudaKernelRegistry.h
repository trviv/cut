#pragma once

#include <cstddef>
#include <cstdint>

namespace cut {

/// One CUDA kernel (native .cu or transpiled HLSL), keyed by the normalized
/// hash of its counterpart SPIR-V (spec-constant literals zeroed). Generated
/// into CompiledCudaKernels.cpp.
struct CudaKernelEntry {
  uint64_t hash;       ///< Normalized SPIR-V hash (see cudaNormalizedSpirvHash).
  const char *name;    ///< "{Func}_{Dtype...}" variant stem, e.g. "Add_Float32_Float32".
  const char *source;  ///< CUDA C++ kernel source (native .cu or transpiled HLSL).
  const char *entry;   ///< extern "C" kernel entry-point symbol name.
  const char *defines; ///< Space-separated KEY=VAL NVRTC defines ("" for transpiled).
  bool native;         ///< True for hand-authored .cu kernels, false for transpiled.
};

/// One embedded shared CUDA header (a *.cuh from operators/impl), passed to
/// every NVRTC compile alongside the prelude and enum header.
struct CudaKernelHeader {
  const char *name;    ///< Include name, e.g. "MatMulCommon.cuh".
  const char *source;  ///< Header contents.
};

/// Looks up a kernel by normalized SPIR-V hash; native entries win over
/// transpiled unless CUT_CUDA_KERNELS=transpiled. Null if absent.
const CudaKernelEntry *lookupCudaKernelByHash(uint64_t hash);

/// Looks up a kernel by exact variant-stem name (same native preference).
const CudaKernelEntry *lookupCudaKernelByName(const char *name);

/// Number of registered CUDA kernels (native + transpiled tables).
size_t cudaKernelCount();

/// Number of embedded shared .cuh headers.
size_t cudaKernelHeaderCount();

/// Returns the embedded header at @p index; null if out of range.
const CudaKernelHeader *cudaKernelHeader(size_t index);

/// The shared CUDA prelude source (vector-type operators + HLSL intrinsics).
const char *cudaPreludeSource();

/// The shared operator-enum header source (ComputeOpsShared.h contents).
const char *cudaEnumsSource();

} // namespace cut
