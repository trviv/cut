#pragma once

#include <ComputeStructs.h>
#include <VulkanCommon.h>

namespace cut {

/**
 * Vulkan implementation of CommandBuffer.
 * Records compute dispatches as Vulkan commands for GPU execution.
 */
class VulkanCommandBuffer : public CommandBuffer {
public:
  /**
   * Constructs a VulkanCommandBuffer with the necessary Vulkan resources.
   * @param device The Vulkan logical device.
   * @param commandPool The command pool to allocate command buffers from.
   * @param queue The compute queue for submissions.
   */
  VulkanCommandBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue);

  ~VulkanCommandBuffer() override = default;

protected:
  /**
   * Vulkan-specific implementation for encoding a dispatch.
   * Records the dispatch as Vulkan commands in the command buffer.
   * @param dispatch Const reference to the compute dispatch being encoded.
   */
  void encodeImpl(const ComputeDispatch &dispatch) override;

private:
  VkDevice device_;
  VkCommandPool commandPool_;
  VkQueue queue_;
  VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
};

} // namespace cut
