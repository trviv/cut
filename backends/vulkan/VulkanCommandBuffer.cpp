#include <ComputeCommon.h>
#include <VulkanCommandBuffer.h>
#include <VulkanContainers.h>

#include <optional>
#include <unordered_map>

namespace cut {

/// Converts a BindingType to its corresponding VkDescriptorType.
/// Returns std::nullopt for types that don't map to descriptor types (e.g.
/// PushConstant).
std::optional<VkDescriptorType> toVkDescriptorType(BindingType type) {
  switch (type) {
  case BindingType::Sampler:
    return VK_DESCRIPTOR_TYPE_SAMPLER;
  case BindingType::UniformBuffer:
    return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  case BindingType::StorageBuffer:
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  case BindingType::SampledImage:
    return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  case BindingType::StorageImage:
    return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  default:
    return std::nullopt;
  }
}

std::vector<VkDescriptorSetLayoutBinding>
createDescriptorSetLayoutBindings(const std::vector<BindingInfo> &bindings) {
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
  layoutBindings.reserve(bindings.size());

  for (const auto &binding : bindings) {
    auto descriptorType = toVkDescriptorType(binding.type);
    if (!descriptorType) {
      continue;
    }

    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = binding.binding;
    layoutBinding.descriptorCount = 1;
    layoutBinding.descriptorType = *descriptorType;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBinding.pImmutableSamplers = nullptr;

    layoutBindings.push_back(layoutBinding);
  }

  return layoutBindings;
}

/// Calculates descriptor pool sizes from shader bindings across all dispatches.
std::vector<VkDescriptorPoolSize> calculateDescriptorPoolSizes(
    const std::vector<ComputeDispatch> &dispatches,
    VulkanShaderContainer &shaderContainer) {
  std::unordered_map<VkDescriptorType, uint32_t> descriptorTypeCounts;

  for (const auto &dispatch : dispatches) {
    const auto &shaderHandle = dispatch.shader();
    if (!shaderHandle) {
      continue;
    }

    const auto *shaderStruct = shaderContainer.getShader(shaderHandle);
    if (!shaderStruct) {
      continue;
    }

    for (const auto &binding : shaderStruct->reflection.bindings) {
      auto descriptorType = toVkDescriptorType(binding.type);
      if (descriptorType) {
        descriptorTypeCounts[*descriptorType]++;
      }
    }
  }

  std::vector<VkDescriptorPoolSize> poolSizes;
  poolSizes.reserve(descriptorTypeCounts.size());
  for (const auto &[type, count] : descriptorTypeCounts) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = type;
    poolSize.descriptorCount = count;
    poolSizes.push_back(poolSize);
  }

  return poolSizes;
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
  auto poolSizes = calculateDescriptorPoolSizes(dispatches(), shaderContainer_);

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
