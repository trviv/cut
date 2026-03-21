#include "VulkanStaging.h"

namespace cut {

VulkanStaging::VulkanStaging(VkDevice device,
                             uint32_t queueFamilyIndex,
                             CreateStagingFn createFn,
                             DestroyStagingFn destroyFn)
    : createStagingFn_(std::move(createFn)),
      destroyStagingFn_(std::move(destroyFn)), device_(device),
      queueFamilyIndex_(queueFamilyIndex) {
  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = queueFamilyIndex_;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &cmdPool_));

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = cmdPool_;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(device_, &allocInfo, &cmdBuf_));

  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VK_CHECK(vkCreateFence(device_, &fenceInfo, nullptr, &fence_));
}

VulkanStaging::~VulkanStaging() {
  flush();

  if (capacity_ > 0) {
    destroyStagingFn_(staging_);
    capacity_ = 0;
  }

  if (fence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_, fence_, nullptr);
  }
  if (cmdPool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, cmdPool_, nullptr);
  }
}

void *VulkanStaging::reserve(size_t bytes) {
  const size_t aligned = (bytes + kAlignment - 1) & ~(kAlignment - 1);

  if (offset_ + aligned > capacity_) {
    flush();
    size_t newCapacity =
        std::max(offset_ + aligned, std::max(capacity_ * 2, kMinCapacity));
    if (capacity_ > 0) {
      destroyStagingFn_(staging_);
    }
    staging_ = createStagingFn_(newCapacity);
    capacity_ = newCapacity;
  }

  void *ptr = static_cast<char *>(staging_.data) + offset_;
  offset_ += aligned;
  return ptr;
}

void VulkanStaging::recordCopy(VkBuffer dstBuffer,
                               VkDeviceSize dstOffset,
                               VkDeviceSize size,
                               const ComputeHandle &dstHandle) {
  if (!recording_) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmdBuf_, &beginInfo));
    recording_ = true;
  }

  // Source offset is where reserve() placed this allocation
  const size_t srcOffset =
      offset_ - ((size + kAlignment - 1) & ~(kAlignment - 1));

  VkBufferCopy copyRegion = {};
  copyRegion.srcOffset = srcOffset;
  copyRegion.dstOffset = dstOffset;
  copyRegion.size = size;
  vkCmdCopyBuffer(cmdBuf_, staging_.buffer, dstBuffer, 1, &copyRegion);

  dstHandles_.push_back(dstHandle);
}

void VulkanStaging::flush() {
  if (!recording_)
    return;

  VK_CHECK(vkEndCommandBuffer(cmdBuf_));

  VkQueue queue;
  vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuf_;

  VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence_));
  VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));

  dstHandles_.clear();
  offset_ = 0;

  VK_CHECK(vkResetFences(device_, 1, &fence_));
  VK_CHECK(vkResetCommandPool(device_, cmdPool_, 0));

  recording_ = false;
}

void VulkanStaging::releaseStagingMemory() {
  if (capacity_ > 0) {
    destroyStagingFn_(staging_);
    capacity_ = 0;
    offset_ = 0;
  }
}

} // namespace cut
