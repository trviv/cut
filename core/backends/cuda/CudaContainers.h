#pragma once

#include "CudaCommandBuffer.h"
#include <ComputeContainers.h>
#include <CudaStructs.h>

#include <map>
#include <optional>
#include <vector>

namespace cut {

/// Base class for CUDA containers that need the owning context handle.
/// The context is made current before destructive driver calls so that
/// container teardown during shutdown targets the right context.
class CudaContainerBase {
public:
  explicit CudaContainerBase(CUcontext context) : context_(context) {}

protected:
  CUcontext getContext() const { return context_; }

private:
  CUcontext context_;
};

/// Container managing device buffer allocations and their lifecycle.
/// Includes a size-keyed cache that recycles freed device allocations to avoid
/// repeated cuMemAlloc / cuMemFree calls (mirrors the Vulkan buffer cache).
class CudaBufferContainer final : public CudaContainerBase,
                                  public ComputeDataContainer<CudaBufferStruct> {
public:
  explicit CudaBufferContainer(CUcontext context)
      : CudaContainerBase(context) {}

  ~CudaBufferContainer();

  ComputeHandle create(CudaBufferStruct &&structData) {
    if (!structData.isView_) {
      activeMemoryBytes_ += structData.size();
    }
    return ComputeDataContainer::create(std::move(structData));
  }

  const CudaBufferStruct &getBuffer(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle);
  }

  size_t activeMemoryBytes() const { return activeMemoryBytes_; }

  /// Try to acquire a cached device allocation of exactly the given size.
  std::optional<CudaBufferStruct> tryAcquireCached(size_t alignedSize);

  /// Free all cached device allocations.
  void drainCache();

private:
  size_t activeMemoryBytes_ = 0;

  static constexpr size_t kMaxCachedBuffers = 256;
  std::multimap<size_t, CudaBufferStruct> bufferCache_;

  /// Frees a single allocation's device/pinned memory.
  void destroyBufferGPU(CudaBufferStruct &buffer);

  /// Called when a handle's refcount reaches zero. Caches the allocation
  /// instead of freeing it immediately (views are simply dropped).
  void destroyAPIObject(const ComputeHandle &handle) override;
};

/// Container managing translated compute kernels and their lifecycle.
class CudaShaderContainer final : public CudaContainerBase,
                                  public ComputeDataContainer<CudaShaderStruct> {
public:
  explicit CudaShaderContainer(CUcontext context)
      : CudaContainerBase(context) {}

  /// Creates a kernel record from source SPIR-V.
  /// Reflection (bindings, local size, push-constant size) is extracted from
  /// the SPIR-V immediately; module/function compilation is performed by the
  /// shader-translation phase.
  ComputeHandle createShader(const std::vector<uint32_t> &spirvCode);

  /// Returns the kernel function for the given handle (may be null).
  CUfunction getFunction(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle).function;
  }

  /// Returns the shader reflection for the given handle by const reference.
  const ShaderReflection &getReflection(const ComputeHandle &handle) const {
    if (!handle) {
      static const ShaderReflection empty{};
      return empty;
    }
    return ComputeDataContainer::get(handle).reflection;
  }

private:
  /// Unloads the CUDA module backing a shader handle.
  void destroyAPIObject(const ComputeHandle &handle) override;
};

/// Holds all CUDA resource containers needed by command buffers.
struct CudaContainers {
  explicit CudaContainers(CUcontext context)
      : bufferContainer(context), shaderContainer(context) {}

  CudaBufferContainer bufferContainer;
  CudaShaderContainer shaderContainer;
};

/// CUDA implementation of CommandBufferContainer.
/// Pre-creates a pool of CUDA streams and hands out command-buffer wrappers in
/// round-robin fashion.
class CudaCommandBufferContainer final : public CudaContainerBase,
                                         public CommandBufferContainer {
public:
  CudaCommandBufferContainer(CUcontext context,
                             uint32_t maxCommandBuffers,
                             CudaContainers &containers);

  ~CudaCommandBufferContainer();

  ComputeHandle createCommandBuffer() override;

private:
  CudaContainers &containers_;
  std::vector<CUstream> streams_;
  uint32_t nextStreamIndex_ = 0;
};

} // namespace cut
