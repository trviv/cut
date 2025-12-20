#include "VulkanStructs.h"
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

/// Holds all Vulkan structures needed to create a compute pipeline.
struct ComputePipelineCreateData {
  VkComputePipelineCreateInfo pipelineInfo{};
  VkPipelineShaderStageCreateInfo shaderStageInfo{};
  VkPushConstantRange pushConstantRange{};
};

/// Creates compute pipelines for all dispatches based on shader reflection.
/// Returns a vector of VulkanPipelineStruct containing pipeline layouts and
/// compute pipelines.
std::vector<VulkanPipelineStruct> createComputePipelines(
    VkDevice device,
    const std::vector<ComputeDispatch> &dispatches,
    const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts,
    VulkanShaderContainer &shaderContainer) {
  std::vector<VulkanPipelineStruct> pipelines;
  pipelines.reserve(dispatches.size());

  // Accumulate pipeline create data for batch creation
  std::vector<ComputePipelineCreateData> pipelineCreateData;
  pipelineCreateData.reserve(dispatches.size());

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
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    // Use descriptor set layout if available for this dispatch
    if (layoutIndex < descriptorSetLayouts.size()) {
      pipelineLayoutInfo.setLayoutCount = 1;
      pipelineLayoutInfo.pSetLayouts = &descriptorSetLayouts[layoutIndex];
    }

    // Add push constant range if the shader uses push constants
    if (shaderStruct->reflection.pushConstantSize > 0) {
      createData.pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      createData.pushConstantRange.offset = 0;
      createData.pushConstantRange.size =
          shaderStruct->reflection.pushConstantSize;
      pipelineLayoutInfo.pushConstantRangeCount = 1;
      pipelineLayoutInfo.pPushConstantRanges = &createData.pushConstantRange;
    }

    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                    &pipelineStruct.pipelineLayout_));

    // Configure shader stage
    createData.shaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    createData.shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    createData.shaderStageInfo.module = shaderStruct->shader;
    createData.shaderStageInfo.pName = "main";

    // Configure compute pipeline create info
    createData.pipelineInfo.sType =
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    createData.pipelineInfo.layout = pipelineStruct.pipelineLayout_;
    createData.pipelineInfo.stage = createData.shaderStageInfo;

    pipelineCreateData.push_back(createData);
    pipelines.push_back(pipelineStruct);
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

/// Creates descriptor set layouts for all dispatches based on shader
/// reflection.
std::vector<VkDescriptorSetLayout>
createDescriptorSetLayouts(VkDevice device,
                           const std::vector<ComputeDispatch> &dispatches,
                           VulkanShaderContainer &shaderContainer) {
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
  descriptorSetLayouts.reserve(dispatches.size());

  for (const auto &dispatch : dispatches) {
    const auto &shaderHandle = dispatch.shader();
    if (!shaderHandle) {
      continue;
    }

    const auto *shaderStruct = shaderContainer.getShader(shaderHandle);
    if (!shaderStruct) {
      continue;
    }

    // Create descriptor set layout bindings from shader reflection
    auto layoutBindings =
        createDescriptorSetLayoutBindings(shaderStruct->reflection.bindings);

    // Create descriptor set layout (can be empty if no bindings)
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings =
        layoutBindings.empty() ? nullptr : layoutBindings.data();

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                         &descriptorSetLayout));
    descriptorSetLayouts.push_back(descriptorSetLayout);
  }

  return descriptorSetLayouts;
}

/// Calculates descriptor pool sizes from shader bindings across all dispatches.
std::vector<VkDescriptorPoolSize>
calculateDescriptorPoolSizes(const std::vector<ComputeDispatch> &dispatches,
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

VulkanCommandBuffer::VulkanCommandBuffer(
    VkDevice device,
    VkCommandPool commandPool,
    VkQueue queue,
    VulkanBufferContainer &bufferContainer,
    VulkanShaderContainer &shaderContainer,
    VulkanDescriptorPoolContainer &descriptorPoolContainer)
    : device_(device), commandPool_(commandPool), queue_(queue),
      bufferContainer_(bufferContainer), shaderContainer_(shaderContainer),
      descriptorPoolContainer_(descriptorPoolContainer) {
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
  const auto poolSizes =
      calculateDescriptorPoolSizes(dispatches(), shaderContainer_);

  // Create descriptor pool if there are any descriptors to allocate
  if (!poolSizes.empty()) {
    const uint32_t maxSets = static_cast<uint32_t>(dispatches().size());
    auto poolHandle = descriptorPoolContainer_.createPool(poolSizes, maxSets);
    descriptorPoolHandles_.push_back(poolHandle);

    VkDescriptorPool descriptorPool =
        descriptorPoolContainer_.getPool(poolHandle);

    // Create descriptor set layouts for all dispatches
    descriptorSetLayouts_ =
        createDescriptorSetLayouts(device_, dispatches(), shaderContainer_);

    // Allocate all descriptor sets in a single call
    if (!descriptorSetLayouts_.empty()) {
      descriptorSets_.resize(descriptorSetLayouts_.size());

      VkDescriptorSetAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocInfo.descriptorPool = descriptorPool;
      allocInfo.descriptorSetCount =
          static_cast<uint32_t>(descriptorSetLayouts_.size());
      allocInfo.pSetLayouts = descriptorSetLayouts_.data();

      VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo,
                                        descriptorSets_.data()));

      // Create compute pipelines from descriptor set layouts
      pipelines_ = createComputePipelines(
          device_, dispatches(), descriptorSetLayouts_, shaderContainer_);

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
        const auto &shaderHandle = dispatch.shader();
        if (!shaderHandle) {
          continue;
        }

        const auto *shaderStruct = shaderContainer_.getShader(shaderHandle);
        if (!shaderStruct) {
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
          for (const auto &info : shaderStruct->reflection.bindings) {
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
          bufferInfos.push_back(bufferInfo);

          VkWriteDescriptorSet descriptorWrite{};
          descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          descriptorWrite.dstSet = descriptorSets_[dispatchIndex];
          descriptorWrite.dstBinding = binding.index();
          descriptorWrite.dstArrayElement = 0;
          descriptorWrite.descriptorType = *descriptorType;
          descriptorWrite.descriptorCount = 1;
          descriptorWrite.pBufferInfo = &bufferInfos.back();

          descriptorWrites.push_back(descriptorWrite);
        }

        ++dispatchIndex;
      }

      // Update all descriptor sets in a single call
      if (!descriptorWrites.empty()) {
        vkUpdateDescriptorSets(device_,
                               static_cast<uint32_t>(descriptorWrites.size()),
                               descriptorWrites.data(), 0, nullptr);
      }

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
                                pipeline.pipelineLayout_, 0, 1,
                                &descriptorSets_[dispatchIndex], 0, nullptr);

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
    if (pipeline.pipelineLayout_ != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device_, pipeline.pipelineLayout_, nullptr);
    }
  }
  pipelines_.clear();

  // Clean up descriptor set layouts (descriptor sets are freed when pool is
  // destroyed)
  for (auto layout : descriptorSetLayouts_) {
    vkDestroyDescriptorSetLayout(device_, layout, nullptr);
  }
  descriptorSetLayouts_.clear();
}

} // namespace cut
