#pragma once

#include "VulkanCommandBuffer.h"
#include <ComputeContainers.h>
#include <VulkanStructs.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace cut {

class VulkanBufferContainer;
class VulkanShaderContainer;
class VulkanDescriptorContainer;
class VulkanDescriptorSetLayoutContainer;
class VulkanPipelineLayoutContainer;
class VulkanPipelineContainer;

/// Holds references to all Vulkan containers needed by command buffers.
struct VulkanContainerRefs {
  VulkanBufferContainer &bufferContainer;
  VulkanShaderContainer &shaderContainer;
  VulkanDescriptorContainer &descriptorContainer;
  VulkanDescriptorSetLayoutContainer &descriptorSetLayoutContainer;
  VulkanPipelineLayoutContainer &pipelineLayoutContainer;
  VulkanPipelineContainer &pipelineContainer;
};

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
   * @param containers References to all Vulkan containers.
   */
  VulkanCommandBufferContainer(VkDevice device,
                               uint32_t queueFamilyIndex,
                               VulkanContainerRefs containers);

  /// Destroys the command pool and waits for the queue to idle.
  ~VulkanCommandBufferContainer();

  /// Creates a Vulkan-specific command buffer.
  ComputeHandle createCommandBuffer() override {
    return ComputeDataContainer::create(new VulkanCommandBuffer(
        getDevice(), commandPool_, queue_, containers_));
  }

  /// Returns the queue handle.
  VkQueue getQueue() const { return queue_; }

  /// Returns the command pool handle.
  VkCommandPool getCommandPool() const { return commandPool_; }

private:
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VulkanContainerRefs containers_;
};

/// Container managing GPU buffer allocations and their lifecycle.
class VulkanBufferContainer final
    : public VulkanContainerBase,
      public ComputeDataContainer<VulkanBufferStruct> {
public:
  /// Constructs a buffer container.
  VulkanBufferContainer() = default;

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
  void destroyAPIObject(const ComputeHandle &handle) override;
};

/// Container managing shader module allocations and their lifecycle.
class VulkanShaderContainer final
    : public VulkanContainerBase,
      public ComputeDataContainer<VulkanShaderStruct> {
public:
  /// Constructs a shader container.
  VulkanShaderContainer() = default;

  ComputeHandle createShader(VulkanShaderStruct &&structData) {
    return ComputeDataContainer::create(std::move(structData));
  }

  /// Returns the shader struct for the given handle.
  VkShaderModule getShader(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle).shader;
  }

  /// Returns the shader reflection for the given handle, or nullopt if invalid.
  ShaderReflection getReflection(const ComputeHandle &handle) const {
    if (!handle) {
      return {};
    }

    return ComputeDataContainer::get(handle).reflection;
  }

private:
  /// Destroys a shader module and releases its Vulkan resources.
  void destroyAPIObject(const ComputeHandle &handle) override;
};

/// Container managing descriptor pool allocations and their lifecycle.
class VulkanDescriptorContainer final
    : public ComputeDataContainer<VulkanDescriptorStruct>,
      public VulkanContainerBase {
public:
  /// Constructs a descriptor container.
  VulkanDescriptorContainer() = default;

  /**
   * Creates a descriptor pool and allocates descriptor sets.
   * @param poolSizes The descriptor pool sizes.
   * @param descriptorSetLayoutHandles Handles to descriptor set layouts.
   * @param descriptorSetLayouts The Vulkan descriptor set layouts.
   * @return Handle to the created descriptor pool struct.
   */
  ComputeHandle createDescriptorSets(
      const std::vector<VkDescriptorPoolSize> &poolSizes,
      const std::vector<ComputeHandle> &descriptorSetLayoutHandles,
      const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts);

  /**
   * Returns the descriptor sets for the given handle.
   * @param handle Handle to the descriptor pool struct.
   * @return Vector of allocated descriptor sets.
   */
  std::vector<VkDescriptorSet>
  getDescriptorSets(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle).descriptorSets;
  }

private:
  /// Destroys a descriptor pool and releases its Vulkan resources.
  void destroyAPIObject(const ComputeHandle &handle) override;
};

/// Container managing descriptor set layout allocations and their lifecycle.
class VulkanDescriptorSetLayoutContainer final
    : public ComputeDataContainer<VulkanDescriptorSetLayoutStruct>,
      public VulkanContainerBase {
public:
  /// Constructs a descriptor set layout container.
  VulkanDescriptorSetLayoutContainer() = default;

  /**
   * Creates multiple descriptor set layouts from an array of create infos.
   * @param createInfos Array of Vulkan descriptor set layout create infos.
   * @param count Number of create infos in the array.
   * @return Vector of handles to the created descriptor set layouts.
   */
  std::vector<ComputeHandle>
  createLayouts(const VkDescriptorSetLayoutCreateInfo *createInfos,
                size_t count);

  /// Returns the descriptor set layout for the given handle.
  VkDescriptorSetLayout getLayout(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle).layout;
  }

  /**
   * Returns multiple descriptor set layouts for the given handles.
   * @param handles Vector of handles to descriptor set layouts.
   * @return Vector of Vulkan descriptor set layouts.
   */
  std::vector<VkDescriptorSetLayout>
  getLayouts(const std::vector<ComputeHandle> &handles) const;

private:
  /**
   * Creates a descriptor set layout from the given create info.
   * @param createInfo The Vulkan descriptor set layout create info.
   * @return Handle to the created descriptor set layout.
   */
  ComputeHandle createLayout(const VkDescriptorSetLayoutCreateInfo &createInfo);

  /// Destroys a descriptor set layout and releases its Vulkan resources.
  void destroyAPIObject(const ComputeHandle &handle) override;
};

/// Container managing pipeline layout allocations and their lifecycle.
class VulkanPipelineLayoutContainer final
    : public ComputeDataContainer<VulkanPipelineLayoutStruct>,
      public VulkanContainerBase {
public:
  /// Constructs a pipeline layout container.
  VulkanPipelineLayoutContainer() = default;

  /**
   * Creates a pipeline layout from the given create info.
   * @param createInfo The pipeline layout create info containing layout and
   * push constant configuration.
   * @return Handle to the created pipeline layout.
   */
  ComputeHandle createLayout(const VulkanPipelineLayoutCreateInfo &createInfo);

  /**
   * Creates multiple pipeline layouts from an array of create infos.
   * @param createInfos Array of pipeline layout create infos.
   * @param count Number of create infos in the array.
   * @return Vector of handles to the created pipeline layouts.
   */
  std::vector<ComputeHandle>
  createLayouts(const VulkanPipelineLayoutCreateInfo *createInfos,
                size_t count);

  /// Returns the pipeline layout for the given handle.
  VkPipelineLayout getLayout(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle).layout;
  }

  /**
   * Returns multiple pipeline layouts for the given handles.
   * @param handles Vector of handles to pipeline layouts.
   * @return Vector of Vulkan pipeline layouts.
   */
  std::vector<VkPipelineLayout>
  getLayouts(const std::vector<ComputeHandle> &handles) const;

private:
  /// Destroys a pipeline layout and releases its Vulkan resources.
  void destroyAPIObject(const ComputeHandle &handle) override;
};

/// Container managing compute pipeline allocations and their lifecycle.
class VulkanPipelineContainer final
    : public ComputeDataContainer<VulkanPipelineStruct>,
      public VulkanContainerBase {
public:
  /// Constructs a compute pipeline container.
  VulkanPipelineContainer() = default;

  /**
   * Creates multiple compute pipelines from shader stages and pipeline layouts.
   * @param shaderStages Vector of shader stage create infos.
   * @param pipelineLayouts Vector of pipeline layouts.
   * @param pipelineLayoutHandles Vector of handles to pipeline layouts.
   * @return Vector of handles to the created compute pipelines.
   */
  std::vector<ComputeHandle> createPipelines(
      const std::vector<VkPipelineShaderStageCreateInfo> &shaderStages,
      const std::vector<VkPipelineLayout> &pipelineLayouts,
      const std::vector<ComputeHandle> &pipelineLayoutHandles);

  /// Returns the compute pipeline for the given handle.
  VkPipeline getPipeline(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle).computePipeline;
  }

  /// Returns the pipeline layout handle for the given pipeline handle.
  ComputeHandle getPipelineLayoutHandle(const ComputeHandle &handle) const {
    return ComputeDataContainer::get(handle).pipelineLayoutHandle;
  }

  /**
   * Returns multiple compute pipelines for the given handles.
   * @param handles Vector of handles to compute pipelines.
   * @return Vector of Vulkan compute pipelines.
   */
  std::vector<VkPipeline>
  getPipelines(const std::vector<ComputeHandle> &handles) const;

private:
  /// Destroys a compute pipeline and releases its Vulkan resources.
  void destroyAPIObject(const ComputeHandle &handle) override;
};

} // namespace cut
