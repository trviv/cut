#include "VulkanStructs.h"
#include "vulkan/vulkan_core.h"
#include <ComputeCommon.h>
#include <VulkanCommandBuffer.h>
#include <VulkanContainers.h>

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

/// Result of descriptor set creation.
struct DescriptorSetResult {
  std::vector<ComputeHandle> layoutHandles;
  ComputeHandle descriptorsHandle;
};

/// Creates descriptor set layouts, pool, and allocates descriptor sets.
DescriptorSetResult
createDescriptorSets(const std::vector<ComputeDispatch> &dispatches,
                     VulkanShaderContainer &shaderContainer,
                     VulkanDescriptorSetLayoutContainer &layoutContainer,
                     VulkanDescriptorContainer &descriptorContainer) {
  DescriptorSetResult result;

  if (dispatches.empty()) {
    return result;
  }

  std::unordered_map<VkDescriptorType, uint32_t> descriptorTypeCounts;

  // Collect all layout bindings (must stay alive until createLayouts)
  std::vector<std::vector<VkDescriptorSetLayoutBinding>> allLayoutBindings;
  allLayoutBindings.reserve(dispatches.size());

  std::vector<VkDescriptorSetLayoutCreateInfo> layoutCreateInfos;
  layoutCreateInfos.reserve(dispatches.size());

  for (const auto &dispatch : dispatches) {
    const auto &reflection = shaderContainer.getReflection(dispatch.shader());

    // Create layout bindings for this dispatch
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.reserve(reflection.bindings.size());

    for (const auto &binding : reflection.bindings) {
      auto descriptorType = toVkDescriptorType(binding.type);
      if (!descriptorType) {
        continue;
      }

      descriptorTypeCounts[*descriptorType]++;

      VkDescriptorSetLayoutBinding layoutBinding{};
      layoutBinding.binding = binding.binding;
      layoutBinding.descriptorCount = 1;
      layoutBinding.descriptorType = *descriptorType;
      layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      layoutBinding.pImmutableSamplers = nullptr;

      layoutBindings.emplace_back(layoutBinding);
    }

    allLayoutBindings.emplace_back(std::move(layoutBindings));
    const auto &bindings = allLayoutBindings.back();

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.empty() ? nullptr : bindings.data();

    layoutCreateInfos.emplace_back(layoutInfo);
  }

  // Create all descriptor set layouts
  result.layoutHandles = layoutContainer.createLayouts(
      layoutCreateInfos.data(), layoutCreateInfos.size());

  // Convert descriptor type counts to pool sizes
  std::vector<VkDescriptorPoolSize> poolSizes;
  poolSizes.reserve(descriptorTypeCounts.size());
  for (const auto &[type, count] : descriptorTypeCounts) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = type;
    poolSize.descriptorCount = count;
    poolSizes.emplace_back(poolSize);
  }

  // Create descriptor pool and allocate sets
  if (!poolSizes.empty()) {
    const auto descriptorSetLayouts =
        layoutContainer.getLayouts(result.layoutHandles);

    result.descriptorsHandle = descriptorContainer.createDescriptorSets(
        poolSizes, result.layoutHandles, descriptorSetLayouts);
  }

  return result;
}

/// Creates compute pipelines for all dispatches based on shader reflection.
/// Returns vector of pipeline handles.
std::vector<ComputeHandle>
createComputePipelines(const std::vector<ComputeDispatch> &dispatches,
                       const std::vector<ComputeHandle> &layoutHandles,
                       VulkanShaderContainer &shaderContainer,
                       VulkanDescriptorSetLayoutContainer &layoutContainer,
                       VulkanPipelineLayoutContainer &pipelineLayoutContainer,
                       VulkanPipelineContainer &pipelineContainer) {
  if (dispatches.empty()) {
    return {};
  }

  // Pre-fetch VkDescriptorSetLayout values for pipeline creation
  const auto &descriptorSetLayouts = layoutContainer.getLayouts(layoutHandles);

  // Collect all pipeline layout create infos
  std::vector<VulkanPipelineLayoutCreateInfo> pipelineLayoutCreateInfos;
  pipelineLayoutCreateInfos.reserve(dispatches.size());

  for (size_t i = 0; i < dispatches.size(); ++i) {
    const auto &reflection =
        shaderContainer.getReflection(dispatches[i].shader());

    VulkanPipelineLayoutCreateInfo createInfo{};
    createInfo.createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    createInfo.descriptorSetLayoutHandle = layoutHandles[i];
    createInfo.createInfo.setLayoutCount = 1;
    createInfo.createInfo.pSetLayouts = &descriptorSetLayouts[i];

    // Add push constant range if the shader uses push constants
    if (reflection.pushConstantSize > 0) {
      createInfo.pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      createInfo.pushConstantRange.offset = 0;
      createInfo.pushConstantRange.size = reflection.pushConstantSize;
      createInfo.createInfo.pushConstantRangeCount = 1;
      createInfo.createInfo.pPushConstantRanges = &createInfo.pushConstantRange;
    }

    pipelineLayoutCreateInfos.emplace_back(createInfo);
  }

  // Create all pipeline layouts in a single call
  auto pipelineLayoutHandles = pipelineLayoutContainer.createLayouts(
      pipelineLayoutCreateInfos.data(), pipelineLayoutCreateInfos.size());

  // Get all VkPipelineLayout values for compute pipeline creation
  const auto vkPipelineLayouts =
      pipelineLayoutContainer.getLayouts(pipelineLayoutHandles);

  // Build shader stage create infos
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
  shaderStages.reserve(dispatches.size());

  for (size_t i = 0; i < dispatches.size(); ++i) {
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderContainer.getShader(dispatches[i].shader());
    stageInfo.pName = "main";
    shaderStages.emplace_back(stageInfo);
  }

  // Create pipelines using the container
  return pipelineContainer.createPipelines(shaderStages, vkPipelineLayouts,
                                           pipelineLayoutHandles);
}

VulkanCommandBuffer::VulkanCommandBuffer(
    VkDevice device,
    VkCommandPool commandPool,
    VkQueue queue,
    VulkanBufferContainer &bufferContainer,
    VulkanShaderContainer &shaderContainer,
    VulkanDescriptorContainer &descriptorContainer,
    VulkanDescriptorSetLayoutContainer &descriptorSetLayoutContainer,
    VulkanPipelineLayoutContainer &pipelineLayoutContainer,
    VulkanPipelineContainer &pipelineContainer)
    : device_(device), commandPool_(commandPool), queue_(queue),
      bufferContainer_(bufferContainer), shaderContainer_(shaderContainer),
      descriptorContainer_(descriptorContainer),
      descriptorSetLayoutContainer_(descriptorSetLayoutContainer),
      pipelineLayoutContainer_(pipelineLayoutContainer),
      pipelineContainer_(pipelineContainer) {
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
  // Create descriptor set layouts, pool, and allocate descriptor sets
  auto descriptorResult =
      createDescriptorSets(dispatches(), shaderContainer_,
                           descriptorSetLayoutContainer_, descriptorContainer_);
  auto descriptorSetLayoutHandles = std::move(descriptorResult.layoutHandles);
  descriptorsHandle_ = std::move(descriptorResult.descriptorsHandle);

  if (!descriptorSetLayoutHandles.empty() && descriptorsHandle_) {
    const auto &descriptorSets =
        descriptorContainer_.getDescriptorSets(descriptorsHandle_);

    // Create compute pipelines from descriptor set layouts
    pipelineHandles_ =
        createComputePipelines(dispatches(), descriptorSetLayoutHandles,
                               shaderContainer_, descriptorSetLayoutContainer_,
                               pipelineLayoutContainer_, pipelineContainer_);

    // Update descriptor sets with buffer bindings from dispatches
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    std::vector<VkDescriptorBufferInfo> bufferInfos;

    // Pre-allocate buffer infos to keep them alive during
    // vkUpdateDescriptorSets
    size_t totalBindings = 0;
    for (const auto &dispatch : dispatches()) {
      totalBindings += dispatch.bindings().size();
    }
    bufferInfos.reserve(totalBindings);

    size_t dispatchIndex = 0;
    for (const auto &dispatch : dispatches()) {
      const auto &reflection =
          shaderContainer_.getReflection(dispatch.shader());

      // Process each binding in the dispatch
      for (const auto &binding : dispatch.bindings()) {
        if (!binding.isHandle()) {
          // Skip data bindings (push constants) for now
          continue;
        }

        // Find the corresponding binding info from shader reflection
        const BindingInfo *bindingInfo = nullptr;
        for (const auto &info : reflection.bindings) {
          if (info.binding == binding.index()) {
            bindingInfo = &info;
            break;
          }
        }

        if (!bindingInfo) {
          continue;
        }

        auto descriptorType = toVkDescriptorType(bindingInfo->type);
        if (!descriptorType) {
          continue;
        }

        // Get the buffer from the handle
        const auto &bufferStruct =
            bufferContainer_.getBuffer(binding.getHandle());

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = bufferStruct.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = bufferStruct.size;
        bufferInfos.emplace_back(bufferInfo);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSets[dispatchIndex];
        descriptorWrite.dstBinding = binding.index();
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = *descriptorType;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfos.back();

        descriptorWrites.emplace_back(descriptorWrite);
      }

      ++dispatchIndex;
    }

    // Update all descriptor sets in a single call
    if (!descriptorWrites.empty()) {
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(descriptorWrites.size()),
                             descriptorWrites.data(), 0, nullptr);
    }

    // Pre-fetch VkPipeline and VkPipelineLayout values for command recording
    const auto vkPipelines = pipelineContainer_.getPipelines(pipelineHandles_);
    std::vector<VkPipelineLayout> pipelineLayouts;
    pipelineLayouts.reserve(pipelineHandles_.size());
    for (const auto &handle : pipelineHandles_) {
      auto layoutHandle = pipelineContainer_.getPipelineLayoutHandle(handle);
      pipelineLayouts.emplace_back(
          pipelineLayoutContainer_.getLayout(layoutHandle));
    }

    // Record commands: bind pipelines, descriptor sets, and dispatch
    dispatchIndex = 0;
    for (const auto &dispatch : dispatches()) {
      const auto &shaderHandle = dispatch.shader();
      if (!shaderHandle) {
        continue;
      }

      if (dispatchIndex >= pipelineHandles_.size()) {
        break;
      }

      // Bind the compute pipeline
      vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                        vkPipelines[dispatchIndex]);

      // Bind the descriptor set
      vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelineLayouts[dispatchIndex], 0, 1,
                              &descriptorSets[dispatchIndex], 0, nullptr);

      // Dispatch compute work
      const auto &tgSize = dispatch.threadgroupSize();
      vkCmdDispatch(commandBuffer_, tgSize.tgSizeX, tgSize.tgSizeY,
                    tgSize.tgSizeZ);

      ++dispatchIndex;
    }

    // Add memory barrier to ensure compute writes are visible to host reads
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memoryBarrier, 0,
                         nullptr, 0, nullptr);
  }

  // End command buffer recording
  VK_CHECK(vkEndCommandBuffer(commandBuffer_));
}

void VulkanCommandBuffer::submit() {
  // Submit to queue
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer_;

  VK_CHECK(vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE));
  VK_CHECK(vkQueueWaitIdle(queue_));

  // Clear pipeline handles (container manages cleanup)
  pipelineHandles_.clear();

  // Clean up descriptor pool handle
  descriptorsHandle_ = ComputeHandle();
}

} // namespace cut
