#pragma once

#include "VulkanStructs.h"
#include <ComputeStructs.h>
#include <VulkanCommon.h>

#include <vector>

namespace cut {

class VulkanBufferContainer;
class VulkanShaderContainer;
class VulkanDescriptorContainer;
class VulkanDescriptorSetLayoutContainer;
class VulkanPipelineLayoutContainer;

/**
 * Vulkan implementation of CommandBuffer.
 * Records compute dispatches as Vulkan commands for GPU execution.
 */
class VulkanCommandBuffer final : public CommandBuffer {
public:
  /**
   * Constructs a VulkanCommandBuffer with the necessary Vulkan resources.
   * @param device The Vulkan logical device.
   * @param commandPool The command pool to allocate command buffers from.
   * @param queue The compute queue for submissions.
   * @param bufferContainer Reference to buffer container for accessing buffers.
   * @param shaderContainer Reference to shader container for accessing shader
   * reflection data.
   * @param descriptorContainer Reference to descriptor container for
   * creating descriptor pools and sets.
   * @param descriptorSetLayoutContainer Reference to descriptor set layout
   * container for creating descriptor set layouts.
   * @param pipelineLayoutContainer Reference to pipeline layout container for
   * creating pipeline layouts.
   */
  VulkanCommandBuffer(
      VkDevice device,
      VkCommandPool commandPool,
      VkQueue queue,
      VulkanBufferContainer &bufferContainer,
      VulkanShaderContainer &shaderContainer,
      VulkanDescriptorContainer &descriptorContainer,
      VulkanDescriptorSetLayoutContainer &descriptorSetLayoutContainer,
      VulkanPipelineLayoutContainer &pipelineLayoutContainer);

  ~VulkanCommandBuffer() override = default;

  /**
   * Begins recording commands to this command buffer.
   * Calls vkBeginCommandBuffer with one-time submit flag.
   */
  void begin() override;

  /**
   * Ends recording commands to this command buffer.
   * Calls vkEndCommandBuffer.
   */
  void end() override;

  /**
   * Vulkan-specific implementation for submitting the command buffer.
   * Submits the command buffer to the queue and waits for completion.
   */
  void submit() override;

private:
  VkDevice device_;
  VkCommandPool commandPool_;
  VkQueue queue_;
  VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
  VulkanBufferContainer &bufferContainer_;
  VulkanShaderContainer &shaderContainer_;
  VulkanDescriptorContainer &descriptorContainer_;
  VulkanDescriptorSetLayoutContainer &descriptorSetLayoutContainer_;
  VulkanPipelineLayoutContainer &pipelineLayoutContainer_;

  /// Active descriptor pool handle created during end().
  ComputeHandle descriptorsHandle_;

  /// Pipelines created during end(), cleaned up in submit().
  std::vector<VulkanPipelineStruct> pipelines_;

  /// Pipeline layout handles created during end(), cleaned up in submit().
  std::vector<ComputeHandle> pipelineLayoutHandles_;
};

} // namespace cut
