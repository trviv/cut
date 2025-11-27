#pragma once

#include <VulkanStructs.h>

namespace cut {

/// Container managing GPU buffer allocations and their lifecycle.
class VulkanBufferContainer final : public ComputeContainer<> {
public:
  /// Constructs a buffer container with a unique type identifier.
  VulkanBufferContainer() : ComputeContainer(101) {}
  CONTAINER_METHODS_IMPL(VulkanBufferStruct);

public:
  IF_VMA_ENABLED_THEN(VmaAllocator allocator);
  IF_VMA_DISABLED_THEN(VkDevice device);

private:
  friend class VulkanCompute;

  /// Destroys a buffer and frees its associated GPU memory.
  void destroy(HandleData &data) override;
};

/// Container managing shader module allocations and their lifecycle.
class VulkanShaderContainer final : public ComputeContainer<> {
public:
  /// Constructs a shader container with a unique type identifier.
  VulkanShaderContainer() : ComputeContainer(102) {}
  CONTAINER_METHODS_IMPL(VulkanShaderStruct);

public:
  VkDevice device;

private:
  friend class VulkanCompute;

  /// Destroys a shader module and releases its Vulkan resources.
  void destroy(HandleData &data) override;
};

} // namespace cut
