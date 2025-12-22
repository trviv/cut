#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeInterface.h>
#include <VulkanCommon.h>

#include <string_view>
#include <vector>

namespace cut {

/// Pairs a Vulkan physical device with its selected compute queue family index.
struct PhysicalDeviceAndQueueIndex {
  VkPhysicalDevice physicalDevice;
  uint32_t queueIndex;
};

/// Configuration options for initializing a Vulkan compute context.
struct VulkanContextConfig {
  VkPhysicalDeviceType preferredType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  uint32_t maxCommandBuffers = 16;
};

/// Represents a GPU buffer with its memory allocation and mapping state.
/// Supports both VMA-managed and manual memory allocation strategies.
struct VulkanBufferStruct {
  static constexpr std::string_view Name = "VulkanBufferStruct";

  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  VkDeviceSize offset = 0;
  void *mappedData = nullptr;
  bool isCoherent = false;

  // VMA dependent members
  IF_VMA_ENABLED_THEN(VmaAllocation allocation = VK_NULL_HANDLE);
  IF_VMA_DISABLED_THEN(VkDeviceMemory memory = VK_NULL_HANDLE);
};

/// Wrapper for a compiled SPIR-V shader module with reflection data.
struct VulkanShaderStruct {
  static constexpr std::string_view Name = "VulkanShaderStruct";

  VkShaderModule shader = VK_NULL_HANDLE;
  ShaderReflection reflection; ///< Binding information from SPIR-V reflection.
};

/// Wrapper for a Vulkan descriptor pool and its allocated descriptor sets.
struct VulkanDescriptorStruct {
  static constexpr std::string_view Name = "VulkanDescriptorStruct";

  VkDescriptorPool pool = VK_NULL_HANDLE;
  std::vector<ComputeHandle> descriptorSetLayoutHandles;
  std::vector<VkDescriptorSet> descriptorSets;
};

/// Wrapper for a Vulkan descriptor set layout.
struct VulkanDescriptorSetLayoutStruct {
  static constexpr std::string_view Name = "VulkanDescriptorSetLayoutStruct";

  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  std::vector<VkDescriptorSetLayoutBinding> bindings;
};

/// Wrapper for a Vulkan pipeline layout.
struct VulkanPipelineLayoutStruct {
  static constexpr std::string_view Name = "VulkanPipelineLayoutStruct";

  VkPipelineLayout layout = VK_NULL_HANDLE;
};

/// Create info for a Vulkan pipeline layout.
struct VulkanPipelineLayoutCreateInfo {
  VkPipelineLayoutCreateInfo createInfo;
  VkPushConstantRange pushConstantRange;
};

/// Contains the pipeline layout and compute pipeline for shader execution.
struct VulkanPipelineStruct {
  static constexpr std::string_view Name = "VulkanPipelineStruct";

  ComputeHandle pipelineLayoutHandle;
  VkPipeline computePipeline = VK_NULL_HANDLE;
};

} // namespace cut
