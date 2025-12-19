#pragma once

#include <ComputeStructs.h>
#include <VulkanCommon.h>

namespace cut {

class VulkanShaderContainer;

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
   * @param shaderContainer Reference to shader container for accessing shader
   * reflection data.
   */
  VulkanCommandBuffer(VkDevice device,
                      VkCommandPool commandPool,
                      VkQueue queue,
                      VulkanShaderContainer &shaderContainer);

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
  VulkanShaderContainer &shaderContainer_;
};

} // namespace cut
