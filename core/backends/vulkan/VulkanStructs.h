#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeInterface.h>
#include <VulkanCommon.h>

#include <cstddef>
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
  /// Preferred physical-device index. May be overridden by the
  /// CUT_VULKAN_DEVICE env var. -1 means "auto-select by preferredType".
  int preferredDevice = -1;
  VkPhysicalDeviceType preferredType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  uint32_t maxCommandBuffers = 16;
};

/// Represents a GPU buffer with its memory allocation and mapping state.
/// Supports both VMA-managed and manual memory allocation strategies.
/// Inherits common buffer properties from ComputeBuffer.
struct VulkanBufferStruct : public ComputeBuffer {
  static constexpr std::string_view Name = "VulkanBufferStruct";

  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  bool isCoherent = false;
  bool isView_ = false;        ///< True if this is a view into a parent buffer.
  ComputeHandle parentHandle_; ///< Ref-counted handle keeping the parent alive.

  /// Returns true if this buffer is device-only (not host-visible).
  bool isDeviceOnly() const { return data == nullptr; }

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
  std::vector<VkDescriptorPoolSize> poolSizes;
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
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  VkPipeline computePipeline = VK_NULL_HANDLE;
};

} // namespace cut
