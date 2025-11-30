#pragma once

#include <ComputeContainers.h>
#include <VulkanStructs.h>

namespace cut {

/// Container managing GPU buffer allocations and their lifecycle.
class VulkanBufferContainer final
    : public ComputeDataContainer<VulkanBufferStruct *> {
public:
  /// Constructs a buffer container with a unique type identifier.
  VulkanBufferContainer() : ComputeDataContainer<VulkanBufferStruct *>(101) {}

  ComputeHandle create(VulkanBufferStruct &&structData) {
    return ComputeDataContainer::create(
        new VulkanBufferStruct(std::move(structData)));
  }

private:
  friend class VulkanCompute;

  IF_VMA_ENABLED_THEN(VmaAllocator allocator_);
  IF_VMA_DISABLED_THEN(VkDevice device_);

  /// Destroys a buffer and frees its associated GPU memory.
  void destroy(const ComputeHandle &handle) override;
};

/// Container managing shader module allocations and their lifecycle.
class VulkanShaderContainer final
    : public ComputeDataContainer<VulkanShaderStruct *> {
public:
  /// Constructs a shader container with a unique type identifier.
  VulkanShaderContainer() : ComputeDataContainer<VulkanShaderStruct *>(102) {}

  ComputeHandle create(VulkanShaderStruct &&structData) {
    return ComputeDataContainer::create(
        new VulkanShaderStruct(std::move(structData)));
  }

private:
  friend class VulkanCompute;

  VkDevice device_;

  /// Destroys a shader module and releases its Vulkan resources.
  void destroy(const ComputeHandle &handle) override;
};

} // namespace cut
