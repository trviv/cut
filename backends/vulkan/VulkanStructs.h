#pragma once

#include <ComputeInterface.h>
#include <VulkanCommon.h>

namespace cut {

struct PhysicalDeviceAndQueueIndex {
  VkPhysicalDevice physicalDevice;
  uint32_t queueIndex;
};

struct VulkanContextConfig {
  VkPhysicalDeviceType preferredType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  uint32_t maxCommandBuffers = 16;
};

struct VulkanCommandStruct {
  VkCommandBuffer command = VK_NULL_HANDLE;
};

struct VulkanBufferStruct {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  VkDeviceSize offset = 0;
  void *mappedData = nullptr;
  bool isCoherent = false;

  // VMA dependent members
  IF_VMA_ENABLED_THEN(VmaAllocation allocation = VK_NULL_HANDLE);
  IF_VMA_DISABLED_THEN(VkDeviceMemory memory = VK_NULL_HANDLE);
};

struct VulkanShaderStruct {
  VkShaderModule shader = VK_NULL_HANDLE;
};

class VulkanDispatchStruct {
public:
  /// Binds a compute shader to this dispatch configuration.
  void bindShader(const ComputeHandle shaaderHandle);
  /// Binds a resource (buffer) to a descriptor binding index.
  void bindResource(const ComputeHandle resourceHandle, uint32_t index);

private:
  struct Binding {
    uint32_t index;
    ComputeHandle handle;
  };

  ComputeHandle shaader_;
  std::vector<Binding> bindings_;

  std::vector<VkDescriptorSetLayoutBinding> descSetLayoutBindings_;
};

struct VulkanDescriptorStruct {
  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
};

struct VulkanPipelineStruct {
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline computePipeline_ = VK_NULL_HANDLE;
};

} // namespace cut
