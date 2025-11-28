#pragma once

#include <ComputeContainers.h>
#include <VulkanStructs.h>

namespace cut {

/// Container managing GPU buffer allocations and their lifecycle.
class VulkanBufferContainer final : public ComputeDataContainer<void *> {
public:
  /// Constructs a buffer container with a unique type identifier.
  VulkanBufferContainer() : ComputeDataContainer<void *>(101) {}

  ComputeHandle createHandle(VulkanBufferStruct &&structData) {
    return ComputeDataContainer<void *>::createHandle(
        new VulkanBufferStruct(std::move(structData)));
  }

  VulkanBufferStruct *get(const ComputeHandle &handle) {
    return data(handle).template get<VulkanBufferStruct *>();
  }

  const VulkanBufferStruct *get(const ComputeHandle &handle) const {
    return data(handle).template get<const VulkanBufferStruct *>();
  }

private:
  friend class VulkanCompute;

  IF_VMA_ENABLED_THEN(VmaAllocator allocator_);
  IF_VMA_DISABLED_THEN(VkDevice device_);

  /// Destroys a buffer and frees its associated GPU memory.
  void destroy(size_t id) override;
};

/// Container managing shader module allocations and their lifecycle.
class VulkanShaderContainer final : public ComputeDataContainer<void *> {
public:
  /// Constructs a shader container with a unique type identifier.
  VulkanShaderContainer() : ComputeDataContainer<void *>(102) {}

  ComputeHandle createHandle(VulkanShaderStruct &&structData) {
    return ComputeDataContainer<void *>::createHandle(
        new VulkanShaderStruct(std::move(structData)));
  }

  VulkanShaderStruct *get(const ComputeHandle &handle) {
    return data(handle).template get<VulkanShaderStruct *>();
  }

  const VulkanShaderStruct *get(const ComputeHandle &handle) const {
    return data(handle).template get<const VulkanShaderStruct *>();
  }

private:
  friend class VulkanCompute;

  VkDevice device_;

  /// Destroys a shader module and releases its Vulkan resources.
  void destroy(size_t id) override;
};

} // namespace cut
