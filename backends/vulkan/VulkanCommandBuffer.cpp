#include <ComputeCommon.h>
#include <VulkanCommandBuffer.h>
#include <VulkanContainers.h>

#include <unordered_map>

namespace cut {

std::vector<VkDescriptorSetLayoutBinding>
createDescriptorSetLayoutBindings(const std::vector<BindingInfo> &bindings) {
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
  layoutBindings.reserve(bindings.size());

  for (const auto &binding : bindings) {
    // Skip push constants - they don't go in descriptor set layouts
    if (binding.type == BindingType::PushConstant) {
      continue;
    }

    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = binding.binding;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBinding.pImmutableSamplers = nullptr;

    switch (binding.type) {
    case BindingType::Sampler:
      layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      break;
    case BindingType::UniformBuffer:
      layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      break;
    case BindingType::StorageBuffer:
      layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      break;
    case BindingType::SampledImage:
      layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      break;
    case BindingType::StorageImage:
      layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      break;
    default:
      continue;
    }

    layoutBindings.push_back(layoutBinding);
  }

  return layoutBindings;
}

VulkanCommandBuffer::VulkanCommandBuffer(VkDevice device,
                                         VkCommandPool commandPool,
                                         VkQueue queue,
                                         VulkanShaderContainer &shaderContainer)
    : device_(device), commandPool_(commandPool), queue_(queue),
      shaderContainer_(shaderContainer) {
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
  // Calculate descriptor pool sizes based on all dispatch shader bindings
  std::unordered_map<VkDescriptorType, uint32_t> descriptorTypeCounts;

  for (const auto &dispatch : dispatches()) {
    const auto &shaderHandle = dispatch.shader();
    if (!shaderHandle) {
      continue;
    }

    // Get shader reflection data
    const auto *shaderStruct = shaderContainer_.getShader(shaderHandle);
    if (!shaderStruct) {
      continue;
    }

    // Count descriptor types from shader bindings
    for (const auto &binding : shaderStruct->reflection.bindings) {
      if (binding.type == BindingType::PushConstant) {
        continue;
      }

      VkDescriptorType descriptorType;
      switch (binding.type) {
      case BindingType::Sampler:
        descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        break;
      case BindingType::UniformBuffer:
        descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        break;
      case BindingType::StorageBuffer:
        descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        break;
      case BindingType::SampledImage:
        descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        break;
      case BindingType::StorageImage:
        descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        break;
      default:
        continue;
      }

      descriptorTypeCounts[descriptorType]++;
    }
  }

  // Build descriptor pool sizes from accumulated counts
  std::vector<VkDescriptorPoolSize> poolSizes;
  poolSizes.reserve(descriptorTypeCounts.size());
  for (const auto &[type, count] : descriptorTypeCounts) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = type;
    poolSize.descriptorCount = count;
    poolSizes.push_back(poolSize);
  }

  // TODO: Create descriptor pool and sets using poolSizes

  // Submit to queue
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer_;

  vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue_);
}

} // namespace cut
