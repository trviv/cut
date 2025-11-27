#include <VulkanCompute.h>
#include <VulkanContainers.h>

namespace cut {

void VulkanBufferContainer::destroy(HandleData &data) {
  auto *bufferPtr = get(data);
  if (bufferPtr == nullptr) {
    // Skip null handle
    return;
  }

#if CUT_USE_VMA
  if (bufferPtr->mappedData != nullptr) {
    vmaUnmapMemory(allocator, bufferPtr->allocation);
  }

  if (bufferPtr->buffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(allocator, bufferPtr->buffer, bufferPtr->allocation);
  }
#else
  if (bufferPtr->mappedData != nullptr) {
    vkUnmapMemory(device_, bufferPtr->memory);
  }

  if (bufferPtr->buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, bufferPtr->buffer, nullptr);
  }

  if (bufferPtr->memory != VK_NULL_HANDLE) {
    vkFreeMemory(device_, bufferPtr->memory, nullptr);
  }
#endif

  delete bufferPtr;
}

void VulkanShaderContainer::destroy(HandleData &data) {
  const auto shaderData = get(data);
  if (shaderData == nullptr) {
    // Skip null handle
    return;
  }

  vkDestroyShaderModule(device_, shaderData->shader, nullptr);

  delete shaderData;
}

} // namespace cut
