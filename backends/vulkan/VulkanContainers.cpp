#include <VulkanCompute.h>
#include <VulkanContainers.h>

namespace cut {

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

} // namespace cut
