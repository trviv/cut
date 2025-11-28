#pragma once

#include <ComputeContainer.h>
#include <VulkanStructs.h>

namespace cut {

/// Container managing GPU buffer allocations and their lifecycle.
class VulkanBufferContainer final : public ComputeDataContainer<void *> {
public:
  /// Constructs a buffer container with a unique type identifier.
  VulkanBufferContainer() : ComputeDataContainer<void *>(101) {}
  CONTAINER_METHODS_IMPL(VulkanBufferStruct);

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
  CONTAINER_METHODS_IMPL(VulkanShaderStruct);

private:
  friend class VulkanCompute;

  VkDevice device_;

  /// Destroys a shader module and releases its Vulkan resources.
  void destroy(size_t id) override;
};

} // namespace cut
