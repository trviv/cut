#include "VulkanStructs.h"
#include "vulkan/vulkan_core.h"
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

std::vector<VkDescriptorSetLayoutBinding> createDescriptorSetLayoutBindings(
    const std::vector<BindingInfo> &bindings,
    std::unordered_map<VkDescriptorType, uint32_t> &descriptorTypeCounts) {
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
  layoutBindings.reserve(bindings.size());

  for (const auto &binding : bindings) {
    auto descriptorType = toVkDescriptorType(binding.type);
    if (!descriptorType) {
      continue;
    }

    descriptorTypeCounts[*descriptorType]++;

    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = binding.binding;
    layoutBinding.descriptorCount = 1;
    layoutBinding.descriptorType = *descriptorType;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBinding.pImmutableSamplers = nullptr;

    layoutBindings.emplace_back(layoutBinding);
  }

  return layoutBindings;
}

/// Result of descriptor set preparation containing layouts and pool sizes.
struct DescriptorSetPreparationResult {
  std::vector<ComputeHandle> layoutHandles;
  std::vector<VkDescriptorPoolSize> poolSizes;
};

/// Creates descriptor set layouts and calculates pool sizes for all dispatches.
DescriptorSetPreparationResult
prepareDescriptorSets(const std::vector<ComputeDispatch> &dispatches,
                      VulkanShaderContainer &shaderContainer,
                      VulkanDescriptorSetLayoutContainer &layoutContainer) {
  DescriptorSetPreparationResult result;
  result.layoutHandles.reserve(dispatches.size());

  std::unordered_map<VkDescriptorType, uint32_t> descriptorTypeCounts;

  for (const auto &dispatch : dispatches) {
    const auto *reflection = shaderContainer.getReflection(dispatch.shader());
    if (!reflection) {
      continue;
    }

    // Create descriptor set layout bindings and accumulate type counts
    const auto layoutBindings = createDescriptorSetLayoutBindings(
        reflection->bindings, descriptorTypeCounts);

    // Create descriptor set layout (can be empty if no bindings)
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings =
        layoutBindings.empty() ? nullptr : layoutBindings.data();

    result.layoutHandles.emplace_back(layoutContainer.createLayout(layoutInfo));
  }

  // Convert descriptor type counts to pool sizes
  result.poolSizes.reserve(descriptorTypeCounts.size());
  for (const auto &[type, count] : descriptorTypeCounts) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = type;
    poolSize.descriptorCount = count;
    result.poolSizes.emplace_back(poolSize);
  }

  return result;
}

/// Holds all Vulkan structures needed to create a compute pipeline.
struct ComputePipelineCreateData {
  VkComputePipelineCreateInfo pipelineInfo{};
  VkPipelineShaderStageCreateInfo shaderStageInfo{};
  VkPushConstantRange pushConstantRange{};
};

/// Creates compute pipelines for all dispatches based on shader reflection.
/// Returns a vector of VulkanPipelineStruct containing pipeline layout handles
/// and compute pipelines.
std::vector<VulkanPipelineStruct>
createComputePipelines(VkDevice device,
                       const std::vector<ComputeDispatch> &dispatches,
                       const std::vector<ComputeHandle> &layoutHandles,
                       VulkanShaderContainer &shaderContainer,
                       VulkanDescriptorSetLayoutContainer &layoutContainer,
                       VulkanPipelineLayoutContainer &pipelineLayoutContainer,
                       std::vector<ComputeHandle> &pipelineLayoutHandles) {
  std::vector<VulkanPipelineStruct> pipelines;
  pipelines.reserve(dispatches.size());

  // Accumulate pipeline create data for batch creation
  std::vector<ComputePipelineCreateData> pipelineCreateData;
  pipelineCreateData.reserve(dispatches.size());

  // Pre-fetch VkDescriptorSetLayout values for pipeline creation
  const auto descriptorSetLayouts = layoutContainer.getLayouts(layoutHandles);

  size_t layoutIndex = 0;
  for (const auto &dispatch : dispatches) {
    const auto &shaderHandle = dispatch.shader();
    if (!shaderHandle) {
      continue;
    }

    const auto *shaderStruct = shaderContainer.getShader(shaderHandle);
    if (!shaderStruct) {
      continue;
    }

    VulkanPipelineStruct pipelineStruct{};
    ComputePipelineCreateData createData{};

    // Create pipeline layout
    VulkanPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.createInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    // Use descriptor set layout if available for this dispatch
    if (layoutIndex < layoutHandles.size()) {
      pipelineLayoutCreateInfo.descriptorSetLayoutHandle =
          layoutHandles[layoutIndex];
      pipelineLayoutCreateInfo.createInfo.setLayoutCount = 1;
      pipelineLayoutCreateInfo.createInfo.pSetLayouts =
          &descriptorSetLayouts[layoutIndex];
    }

    // Add push constant range if the shader uses push constants
    if (shaderStruct->reflection.pushConstantSize > 0) {
      pipelineLayoutCreateInfo.pushConstantRange.stageFlags =
          VK_SHADER_STAGE_COMPUTE_BIT;
      pipelineLayoutCreateInfo.pushConstantRange.offset = 0;
      pipelineLayoutCreateInfo.pushConstantRange.size =
          shaderStruct->reflection.pushConstantSize;
      pipelineLayoutCreateInfo.createInfo.pushConstantRangeCount = 1;
      pipelineLayoutCreateInfo.createInfo.pPushConstantRanges =
          &pipelineLayoutCreateInfo.pushConstantRange;
    }

    // Create pipeline layout using container
    auto pipelineLayoutHandle =
        pipelineLayoutContainer.createLayout(pipelineLayoutCreateInfo);
    pipelineStruct.pipelineLayoutHandle_ = pipelineLayoutHandle;
    pipelineLayoutHandles.emplace_back(std::move(pipelineLayoutHandle));

    // Configure shader stage
    createData.shaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    createData.shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    createData.shaderStageInfo.module = shaderStruct->shader;
    createData.shaderStageInfo.pName = "main";

    // Configure compute pipeline create info
    createData.pipelineInfo.sType =
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    createData.pipelineInfo.layout =
        pipelineLayoutContainer.getLayout(pipelineStruct.pipelineLayoutHandle_);
    createData.pipelineInfo.stage = createData.shaderStageInfo;

    pipelineCreateData.emplace_back(createData);
    pipelines.emplace_back(pipelineStruct);
    ++layoutIndex;
  }

  // Create all compute pipelines in a single call
  if (!pipelineCreateData.empty()) {
    // Extract pipeline create infos for the batch call
    std::vector<VkComputePipelineCreateInfo> pipelineCreateInfos;
    pipelineCreateInfos.reserve(pipelineCreateData.size());
    for (const auto &data : pipelineCreateData) {
      pipelineCreateInfos.push_back(data.pipelineInfo);
    }

    std::vector<VkPipeline> vkPipelines(pipelineCreateInfos.size());
    VK_CHECK(vkCreateComputePipelines(
        device, VK_NULL_HANDLE,
        static_cast<uint32_t>(pipelineCreateInfos.size()),
        pipelineCreateInfos.data(), nullptr, vkPipelines.data()));

    // Store the created pipelines in the result structs
    for (size_t i = 0; i < vkPipelines.size(); ++i) {
      pipelines[i].computePipeline_ = vkPipelines[i];
    }
  }

  return pipelines;
}

VulkanCommandBuffer::VulkanCommandBuffer(
    VkDevice device,
    VkCommandPool commandPool,
    VkQueue queue,
    VulkanBufferContainer &bufferContainer,
    VulkanShaderContainer &shaderContainer,
    VulkanDescriptorContainer &descriptorContainer,
    VulkanDescriptorSetLayoutContainer &descriptorSetLayoutContainer,
    VulkanPipelineLayoutContainer &pipelineLayoutContainer)
    : device_(device), commandPool_(commandPool), queue_(queue),
      bufferContainer_(bufferContainer), shaderContainer_(shaderContainer),
      descriptorContainer_(descriptorContainer),
      descriptorSetLayoutContainer_(descriptorSetLayoutContainer),
      pipelineLayoutContainer_(pipelineLayoutContainer) {
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
  // Prepare descriptor sets: create layouts and calculate pool sizes
  auto preparation = prepareDescriptorSets(dispatches(), shaderContainer_,
                                           descriptorSetLayoutContainer_);
  auto descriptorSetLayoutHandles = std::move(preparation.layoutHandles);

  // Create descriptor pool and allocate descriptor sets
  if (!preparation.poolSizes.empty() && !descriptorSetLayoutHandles.empty()) {
    const auto descriptorSetLayouts =
        descriptorSetLayoutContainer_.getLayouts(descriptorSetLayoutHandles);

    descriptorPoolHandle_ = descriptorContainer_.createDescriptorSets(
        preparation.poolSizes, descriptorSetLayoutHandles,
        descriptorSetLayouts);
    const auto &descriptorSets =
        descriptorContainer_.getDescriptorSets(descriptorPoolHandle_);

    // Create compute pipelines from descriptor set layouts
    pipelines_ = createComputePipelines(
        device_, dispatches(), descriptorSetLayoutHandles, shaderContainer_,
        descriptorSetLayoutContainer_, pipelineLayoutContainer_,
        pipelineLayoutHandles_);

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
      const auto *reflection =
          shaderContainer_.getReflection(dispatch.shader());
      if (!reflection) {
        continue;
      }

      // Process each binding in the dispatch
      for (const auto &binding : dispatch.bindings()) {
        if (!binding.isHandle()) {
          // Skip data bindings (push constants) for now
          continue;
        }

        // Find the corresponding binding info from shader reflection
        const BindingInfo *bindingInfo = nullptr;
        for (const auto &info : reflection->bindings) {
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

    // Pre-fetch VkPipelineLayout values for command recording
    const auto pipelineLayouts =
        pipelineLayoutContainer_.getLayouts(pipelineLayoutHandles_);

    // Record commands: bind pipelines, descriptor sets, and dispatch
    dispatchIndex = 0;
    for (const auto &dispatch : dispatches()) {
      const auto &shaderHandle = dispatch.shader();
      if (!shaderHandle) {
        continue;
      }

      if (dispatchIndex >= pipelines_.size()) {
        break;
      }

      const auto &pipeline = pipelines_[dispatchIndex];

      // Bind the compute pipeline
      vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline.computePipeline_);

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

  // Clean up pipelines after GPU execution completes
  for (auto &pipeline : pipelines_) {
    if (pipeline.computePipeline_ != VK_NULL_HANDLE) {
      vkDestroyPipeline(device_, pipeline.computePipeline_, nullptr);
    }
  }
  pipelines_.clear();

  // Clean up pipeline layout handles (the container will destroy the layouts)
  pipelineLayoutHandles_.clear();

  // Clean up descriptor set layout handles (the container will destroy the
  // layouts, descriptor sets are freed when pool is destroyed)
  //  descriptorSetLayoutHandles.clear();

  // Clean up descriptor pool handle
  descriptorPoolHandle_ = ComputeHandle();
}

} // namespace cut
