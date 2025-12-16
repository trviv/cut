#pragma once

#include "VulkanCommandBuffer.h"
#include <ComputeContainers.h>
#include <VulkanStructs.h>

#include <memory>
#include <utility>

namespace cut {

/// Vulkan implementation of CommandBufferContainer.
class VulkanCommandBufferContainer final : public CommandBufferContainer {
public:
  /**
   * Constructs a Vulkan command buffer container.
   * Creates the command pool and retrieves the queue internally.
   * @param device The Vulkan logical device.
   * @param queueFamilyIndex The queue family index for command submission.
   */
  VulkanCommandBufferContainer(VkDevice device, uint32_t queueFamilyIndex);

  /// Destroys the command pool and waits for the queue to idle.
  ~VulkanCommandBufferContainer();

  /// Creates a Vulkan-specific command buffer.
  ComputeHandle createCommandBuffer() override {
    return ComputeDataContainer::create(
        new VulkanCommandBuffer(device_, commandPool_, queue_));
  }

  /// Returns the queue handle.
  VkQueue getQueue() const { return queue_; }

  /// Returns the command pool handle.
  VkCommandPool getCommandPool() const { return commandPool_; }

private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
};

/// Container managing GPU buffer allocations and their lifecycle.
class VulkanBufferContainer final
    : public ComputeDataContainer<VulkanBufferStruct> {
public:
  /// Constructs a buffer container with a unique type identifier.
  VulkanBufferContainer() : ComputeDataContainer<VulkanBufferStruct>(101) {}

  ComputeHandle create(VulkanBufferStruct &&structData) {
    return ComputeDataContainer::create(std::move(structData));
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
