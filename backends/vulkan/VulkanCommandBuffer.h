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
class VulkanPipelineContainer;
struct VulkanContainerRefs;

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
   * @param containers References to all Vulkan containers.
   */
  VulkanCommandBuffer(VkDevice device,
                      VkCommandPool commandPool,
                      VkQueue queue,
                      VulkanContainerRefs &containers);

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
   * Submits the command buffer to the queue.
   */
  void submit() override;

  /**
   * Waits for the command buffer to finish execution.
   * Blocks until all submitted commands have completed.
   */
  void wait() override;

  ~VulkanCommandBuffer() override;

private:
  VkDevice device_;
  VkCommandPool commandPool_;
  VkQueue queue_;
  VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  bool submitted_ = false;
  VulkanContainerRefs &containers_;

  /// Active descriptor pool handle created during end().
  ComputeHandle descriptorsHandle_;

  /// Pipeline handles created during end().
  std::vector<ComputeHandle> pipelineHandles_;
};

} // namespace cut
