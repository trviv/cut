#pragma once

#include "VulkanStructs.h"
#include <ComputeStructs.h>
#include <VulkanCommon.h>

#include <vector>

namespace cut {

struct VulkanContainers;

/**
 * Vulkan implementation of CommandBuffer.
 * Records compute dispatches as Vulkan commands for GPU execution.
 */
class VulkanCommandBuffer final : public CommandBuffer {
public:
  /**
   * Constructs a VulkanCommandBuffer with the necessary Vulkan resources.
   * @param device The Vulkan logical device.
   * @param commandBuffer Pre-allocated Vulkan command buffer.
   * @param fence Pre-allocated Vulkan fence for synchronization.
   * @param queue The compute queue for submissions.
   * @param containers Reference to the Vulkan containers struct.
   */
  VulkanCommandBuffer(VkDevice device,
                      VkCommandBuffer commandBuffer,
                      VkFence fence,
                      VkQueue queue,
                      VulkanContainers &containers);

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
  VkQueue queue_;
  VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  bool submitted_ = false;
  VulkanContainers &containers_;

  /// Active descriptor pool handle created during end().
  ComputeHandle descriptorsHandle_;

  /// Pipeline handles created during end().
  std::vector<ComputeHandle> pipelineHandles_;
};

} // namespace cut
