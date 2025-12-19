#include <VulkanCompute.h>
#include <VulkanContainers.h>

namespace cut {

VulkanCommandBufferContainer::VulkanCommandBufferContainer(
    VkDevice device,
    uint32_t queueFamilyIndex,
    VulkanShaderContainer &shaderContainer,
    VulkanDescriptorPoolContainer &descriptorPoolContainer)
    : device_(device), shaderContainer_(shaderContainer),
      descriptorPoolContainer_(descriptorPoolContainer) {
  vkGetDeviceQueue(device_, queueFamilyIndex, 0, &queue_);

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = queueFamilyIndex;

  VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_));
}

VulkanCommandBufferContainer::~VulkanCommandBufferContainer() {
  if (queue_ != VK_NULL_HANDLE) {
    vkQueueWaitIdle(queue_);
  }
  if (commandPool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, commandPool_, nullptr);
  }
}

void VulkanBufferContainer::destroy(const ComputeHandle &handle) {
  auto &buffer = get(handle);

#if CUT_USE_VMA
  if (buffer.mappedData != nullptr) {
    vmaUnmapMemory(allocator, buffer.allocation);
  }

  if (buffer.buffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
  }
#else
  if (buffer.mappedData != nullptr) {
    vkUnmapMemory(device_, buffer.memory);
  }

  if (buffer.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, buffer.buffer, nullptr);
  }

  if (buffer.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device_, buffer.memory, nullptr);
  }
#endif
}

void VulkanShaderContainer::destroy(const ComputeHandle &handle) {
  auto *shaderData = get(handle);
  if (shaderData == nullptr) {
    // Skip null handle
    return;
  }

  vkDestroyShaderModule(device_, shaderData->shader, nullptr);

  delete shaderData;
}

ComputeHandle VulkanDescriptorPoolContainer::createPool(
    const std::vector<VkDescriptorPoolSize> &poolSizes, uint32_t maxSets) {
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = maxSets;

  VulkanDescriptorPoolStruct poolStruct{};
  VK_CHECK(
      vkCreateDescriptorPool(device_, &poolInfo, nullptr, &poolStruct.pool));

  return ComputeDataContainer::create(std::move(poolStruct));
}

void VulkanDescriptorPoolContainer::destroy(const ComputeHandle &handle) {
  auto &poolStruct = get(handle);
  if (poolStruct.pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device_, poolStruct.pool, nullptr);
  }
}

} // namespace cut
