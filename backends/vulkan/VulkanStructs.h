#pragma once

#include <ComputeCommon.h>
#include <ComputeInterface.h>
#include <VulkanCommon.h>

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

/// Wrapper for a Vulkan command buffer used to record GPU commands.
struct VulkanCommandStruct {
  VkCommandBuffer command = VK_NULL_HANDLE;
};

/// Represents a GPU buffer with its memory allocation and mapping state.
/// Supports both VMA-managed and manual memory allocation strategies.
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

/// Wrapper for a compiled SPIR-V shader module with reflection data.
struct VulkanShaderStruct {
  VkShaderModule shader = VK_NULL_HANDLE;
  ShaderReflection reflection; ///< Binding information from SPIR-V reflection.
};

/// Manages compute shader dispatch configuration, including shader and resource
/// bindings.
class VulkanDispatchStruct {
public:
  /// Binds a compute shader to this dispatch configuration.
  void bindShader(const ComputeHandle shaaderHandle);
  /// Binds a resource (buffer) to a descriptor binding index.
  void bindResource(const ComputeHandle resourceHandle, uint32_t index);

private:
  /// Associates a descriptor binding index with a resource handle.
  struct Binding {
    uint32_t index;
    ComputeHandle handle;
  };

  ComputeHandle shaader_;
  std::vector<Binding> bindings_;

  std::vector<VkDescriptorSetLayoutBinding> descSetLayoutBindings_;
};

/// Holds descriptor pool and set for binding resources to shaders.
struct VulkanDescriptorStruct {
  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
};

/// Wrapper for a Vulkan descriptor pool.
struct VulkanDescriptorPoolStruct {
  VkDescriptorPool pool = VK_NULL_HANDLE;
};

/// Contains the pipeline layout and compute pipeline for shader execution.
struct VulkanPipelineStruct {
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline computePipeline_ = VK_NULL_HANDLE;
};

} // namespace cut
