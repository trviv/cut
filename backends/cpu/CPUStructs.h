#pragma once

#include <ComputeCommon.h>

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace cut {

/**
 * CPU kernel function signature.
 * Simple iteration-based execution model for CPU.
 *
 * @param index - flat iteration index (0 to totalIterations-1)
 * @param bindings - pointers to bound buffers (indexed by binding number)
 * @param pushConstants - pointer to push constant data
 */
using CPUKernel = std::function<void(uint32_t index,
                                     const std::vector<void *> &bindings,
                                     const void *pushConstants)>;

/**
 * CPU buffer data structure.
 * Stores a pointer to heap-allocated memory.
 */
struct CPUBufferStruct {
  static constexpr std::string_view Name = "CPUBuffer";

  void *data = nullptr; ///< Pointer to allocated buffer memory.
  size_t size = 0;      ///< Size of the buffer in bytes.
};

/**
 * CPU shader data structure.
 * Stores the kernel function and shader reflection data.
 */
struct CPUShaderStruct {
  static constexpr std::string_view Name = "CPUShader";

  CPUKernel kernel;            ///< User-provided C++ kernel function.
  ShaderReflection reflection; ///< Binding info extracted from SPIR-V.
};

} // namespace cut
