#pragma once

#include "VulkanCommandBuffer.h"
#include <ComputeContainers.h>
#include <VulkanStructs.h>

#include <memory>
#include <utility>
#include <vector>

namespace cut {

class VulkanBufferContainer;
class VulkanShaderContainer;
class VulkanDescriptorPoolContainer;

/// Base class for Vulkan containers that require a device handle.
class VulkanContainerBase {
public:
  /// Sets the Vulkan device handle.
  void setDevice(VkDevice device) { device_ = device; }

protected:
  /// Returns the Vulkan device handle.
  VkDevice getDevice() const { return device_; }

private:
  VkDevice device_ = VK_NULL_HANDLE;
};

/// Vulkan implementation of CommandBufferContainer.
class VulkanCommandBufferContainer final : public VulkanContainerBase,
                                           public CommandBufferContainer {
public:
  /**
   * Constructs a Vulkan command buffer container.
   * Creates the command pool and retrieves the queue internally.
   * @param device The Vulkan logical device.
   * @param queueFamilyIndex The queue family index for command submission.
   * @param bufferContainer Reference to buffer container for accessing buffers.
   * @param shaderContainer Reference to shader container for accessing shader
   * reflection data.
   * @param descriptorPoolContainer Reference to descriptor pool container for
   * creating descriptor pools.
   */
  VulkanCommandBufferContainer(
      VkDevice device,
      uint32_t queueFamilyIndex,
      VulkanBufferContainer &bufferContainer,
      VulkanShaderContainer &shaderContainer,
      VulkanDescriptorPoolContainer &descriptorPoolContainer);

  /// Destroys the command pool and waits for the queue to idle.
  ~VulkanCommandBufferContainer();

  /// Creates a Vulkan-specific command buffer.
  ComputeHandle createCommandBuffer() override {
    return ComputeDataContainer::create(new VulkanCommandBuffer(
        getDevice(), commandPool_, queue_, bufferContainer_, shaderContainer_,
        descriptorPoolContainer_));
  }

  /// Returns the queue handle.
  VkQueue getQueue() const { return queue_; }

  /// Returns the command pool handle.
  VkCommandPool getCommandPool() const { return commandPool_; }

private:
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VulkanBufferContainer &bufferContainer_;
  VulkanShaderContainer &shaderContainer_;
  VulkanDescriptorPoolContainer &descriptorPoolContainer_;
};

/// Container managing GPU buffer allocations and their lifecycle.
class VulkanBufferContainer final
    : public VulkanContainerBase,
      public ComputeDataContainer<VulkanBufferStruct> {
public:
  /// Constructs a buffer container with a unique type identifier.
  VulkanBufferContainer() : ComputeDataContainer<VulkanBufferStruct>(101) {}

  ComputeHandle create(VulkanBufferStruct &&structData) {
    return ComputeDataContainer::create(std::move(structData));
  }

  /// Returns the buffer struct for the given handle.
  const VulkanBufferStruct &getBuffer(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle);
  }

private:
  friend class VulkanCompute;

  IF_VMA_ENABLED_THEN(VmaAllocator allocator_);

  /// Destroys a buffer and frees its associated GPU memory.
  void destroy(const ComputeHandle &handle) override;
};

/// Container managing shader module allocations and their lifecycle.
class VulkanShaderContainer final
    : public VulkanContainerBase,
      public ComputeDataContainer<VulkanShaderStruct *> {
public:
  /// Constructs a shader container with a unique type identifier.
  VulkanShaderContainer() : ComputeDataContainer<VulkanShaderStruct *>(102) {}

  ComputeHandle createShader(VulkanShaderStruct &&structData) {
    return ComputeDataContainer::create(
        new VulkanShaderStruct(std::move(structData)));
  }

  /// Returns the shader struct for the given handle.
  VulkanShaderStruct *getShader(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle);
  }

private:
  /// Destroys a shader module and releases its Vulkan resources.
  void destroy(const ComputeHandle &handle) override;
};

/// Container managing descriptor pool allocations and their lifecycle.
class VulkanDescriptorPoolContainer final
    : public ComputeDataContainer<VulkanDescriptorPoolStruct>,
      public VulkanContainerBase {
public:
  /// Constructs a descriptor pool container with a unique type identifier.
  VulkanDescriptorPoolContainer()
      : ComputeDataContainer<VulkanDescriptorPoolStruct>(103) {}

  /// Creates a descriptor pool with the given pool sizes.
  ComputeHandle createPool(const std::vector<VkDescriptorPoolSize> &poolSizes,
                           uint32_t maxSets);

  /// Returns the descriptor pool for the given handle.
  VkDescriptorPool getPool(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle).pool;
  }

private:
  /// Destroys a descriptor pool and releases its Vulkan resources.
  void destroy(const ComputeHandle &handle) override;
};

/// Container managing descriptor set layout allocations and their lifecycle.
class VulkanDescriptorSetLayoutContainer final
    : public ComputeDataContainer<VulkanDescriptorSetLayoutStruct>,
      public VulkanContainerBase {
public:
  /// Constructs a descriptor set layout container with a unique type identifier.
  VulkanDescriptorSetLayoutContainer()
      : ComputeDataContainer<VulkanDescriptorSetLayoutStruct>(104) {}

  /**
   * Creates a descriptor set layout from the given create info.
   * @param createInfo The Vulkan descriptor set layout create info.
   * @return Handle to the created descriptor set layout.
   */
  ComputeHandle createLayout(const VkDescriptorSetLayoutCreateInfo &createInfo);

  /// Returns the descriptor set layout for the given handle.
  VkDescriptorSetLayout getLayout(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle).layout;
  }

  /// Returns the descriptor set layout struct for the given handle.
  const VulkanDescriptorSetLayoutStruct &
  getLayoutStruct(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle);
  }

private:
  /// Destroys a descriptor set layout and releases its Vulkan resources.
  void destroy(const ComputeHandle &handle) override;
};

} // namespace cut
