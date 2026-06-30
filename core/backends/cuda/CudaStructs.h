#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeInterface.h>
#include <CudaCommon.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cut {

/// Configuration options for initializing a CUDA compute context.
struct CudaContextConfig {
  /// Preferred device ordinal. May be overridden by the CUT_CUDA_DEVICE env
  /// var. -1 means "pick the first compute-capable device".
  int preferredDevice = -1;
  uint32_t maxCommandBuffers = 16;
};

/// Represents a GPU buffer backed by a CUDA Driver API device allocation.
/// Inherits common buffer properties (shape, dtype, sizes) from ComputeBuffer.
///
/// Memory model:
///   - Device-only buffers: devPtr points to device memory, data == nullptr.
///   - Mapped/pinned buffers: data points to page-locked host memory and
///     devPtr is the device-accessible address of that same allocation.
struct CudaBufferStruct : public ComputeBuffer {
  static constexpr std::string_view Name = "CudaBufferStruct";

  CUdeviceptr devPtr = 0;   ///< Device pointer (base of allocation or view).
  size_t offset = 0;        ///< Byte offset into the parent allocation.
  bool isPinned = false;    ///< True if backed by page-locked host memory.
  bool isView_ = false;     ///< True if this is a view into a parent buffer.
  ComputeHandle parentHandle_; ///< Ref-counted handle keeping the parent alive.

  /// Returns true if this buffer is device-only (no host mapping).
  bool isDeviceOnly() const { return data == nullptr; }

  /// Device pointer adjusted for this buffer's offset (for kernel binding).
  CUdeviceptr boundPtr() const { return devPtr + offset; }
};

/// Wrapper for a compute kernel translated to a loadable CUDA module.
///
/// For the initial API-side phase the module/function may be null: the
/// reflection (extracted from the source SPIR-V) is still populated so the
/// command buffer can compute launch geometry. The CUmodule / CUfunction are
/// filled in once the shader-translation phase lands.
struct CudaShaderStruct {
  static constexpr std::string_view Name = "CudaShaderStruct";

  CUmodule module = nullptr;       ///< Loaded PTX/cubin module (may be null).
  CUfunction function = nullptr;   ///< Entry-point kernel (may be null).
  std::string entryName = "main";  ///< Kernel entry-point symbol name.
  ShaderReflection reflection;     ///< Bindings / local size / push-constant size.
};

} // namespace cut
