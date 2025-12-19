#include <VulkanCommandBuffer.h>

namespace cut {

VulkanCommandBuffer::VulkanCommandBuffer(VkDevice device,
                                         VkCommandPool commandPool,
                                         VkQueue queue)
    : device_(device), commandPool_(commandPool), queue_(queue) {
  // Allocate command buffer
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool_;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer_);
}

void VulkanCommandBuffer::begin() {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer_, &beginInfo);
}

void VulkanCommandBuffer::end() {
  vkEndCommandBuffer(commandBuffer_);
}

void VulkanCommandBuffer::submit() {
  // Submit to queue
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer_;

  vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue_);
}

} // namespace cut
