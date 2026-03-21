#include "VulkanStructs.h"
#include "vulkan/vulkan_core.h"
#include <ComputeCommon.h>
#include <VulkanCommandBuffer.h>
#include <VulkanContainers.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

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
  {
    // Collect all layout bindings for each dispatch
    std::vector<std::vector<VkDescriptorSetLayoutBinding>> allLayoutBindings;
    allLayoutBindings.reserve(dispatches.size());

    for (const auto &dispatch : dispatches) {
      if (dispatch.isBarrier() || dispatch.isBufferUpdate()) {
        continue; // Skip non-shader dispatches — no shader or descriptors
      }

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
    }

    // Create all descriptor set layouts
    result.layoutHandles = layoutContainer.createLayouts(allLayoutBindings);
  }

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
    const auto &descriptorSetLayouts =
        layoutContainer.getLayouts(result.layoutHandles);

    result.descriptorsHandle = descriptorContainer.createDescriptorSets(
        poolSizes, result.layoutHandles, descriptorSetLayouts);
  }

  return result;
}

/// Result of compute pipeline creation.
struct ComputePipelineResult {
  std::vector<ComputeHandle> pipelineHandles;
  std::vector<VkPipelineLayout> pipelineLayouts;
};

/// Creates compute pipelines for all dispatches based on shader reflection.
/// Returns pipeline handles and VkPipelineLayouts for command recording.
ComputePipelineResult
createComputePipelines(const std::vector<ComputeDispatch> &dispatches,
                       const std::vector<ComputeHandle> &layoutHandles,
                       VulkanShaderContainer &shaderContainer,
                       VulkanDescriptorSetLayoutContainer &layoutContainer,
                       VulkanPipelineLayoutContainer &pipelineLayoutContainer,
                       VulkanPipelineContainer &pipelineContainer) {
  ComputePipelineResult result;

  if (dispatches.empty()) {
    return result;
  }

  // Pre-fetch VkDescriptorSetLayout values for pipeline creation
  const auto &descriptorSetLayouts = layoutContainer.getLayouts(layoutHandles);

  // Collect all pipeline layout create infos
  std::vector<VulkanPipelineLayoutCreateInfo> pipelineLayoutCreateInfos;
  pipelineLayoutCreateInfos.reserve(dispatches.size());

  size_t layoutIndex = 0;
  for (size_t i = 0; i < dispatches.size(); ++i) {
    if (dispatches[i].isBarrier() || dispatches[i].isBufferUpdate()) {
      continue; // Skip non-shader dispatches — no shader or pipeline
    }

    const auto &reflection =
        shaderContainer.getReflection(dispatches[i].shader());

    VulkanPipelineLayoutCreateInfo createInfo{};
    createInfo.createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    createInfo.createInfo.setLayoutCount = 1;
    createInfo.createInfo.pSetLayouts = &descriptorSetLayouts[layoutIndex];

    // Add push constant range if the shader uses push constants
    if (reflection.pushConstantSize > 0) {
      createInfo.pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      createInfo.pushConstantRange.offset = 0;
      createInfo.pushConstantRange.size = reflection.pushConstantSize;
      createInfo.createInfo.pushConstantRangeCount = 1;
      createInfo.createInfo.pPushConstantRanges = &createInfo.pushConstantRange;
    }

    pipelineLayoutCreateInfos.emplace_back(createInfo);
    // Fix up the pPushConstantRanges pointer to refer to the stored copy's
    // own pushConstantRange member, not the now-dead stack local.
    auto &stored = pipelineLayoutCreateInfos.back();
    if (stored.createInfo.pushConstantRangeCount > 0) {
      stored.createInfo.pPushConstantRanges = &stored.pushConstantRange;
    }
    ++layoutIndex;
  }

  // Create all pipeline layouts in a single call
  const auto &pipelineLayoutHandles =
      pipelineLayoutContainer.createLayouts(pipelineLayoutCreateInfos);

  // Get all VkPipelineLayout values for compute pipeline creation and command
  // recording
  result.pipelineLayouts =
      pipelineLayoutContainer.getLayouts(pipelineLayoutHandles);

  // Build shader stage create infos
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
  shaderStages.reserve(dispatches.size());

  for (size_t i = 0; i < dispatches.size(); ++i) {
    if (dispatches[i].isBarrier() || dispatches[i].isBufferUpdate()) {
      continue; // Skip non-shader dispatches
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderContainer.getShader(dispatches[i].shader());
    stageInfo.pName = "main";
    shaderStages.emplace_back(stageInfo);
  }

  // Create pipelines using the container
  result.pipelineHandles = pipelineContainer.createPipelines(
      shaderStages, result.pipelineLayouts, pipelineLayoutHandles);

  // Get pipeline layouts from the returned pipelines (may be cached)
  // This ensures we use the correct layout for each pipeline
  result.pipelineLayouts.clear();
  result.pipelineLayouts.reserve(result.pipelineHandles.size());
  for (const auto &pipelineHandle : result.pipelineHandles) {
    auto layoutHandle =
        pipelineContainer.getPipelineLayoutHandle(pipelineHandle);
    result.pipelineLayouts.emplace_back(
        pipelineLayoutContainer.getLayouts({layoutHandle})[0]);
  }

  return result;
}

VulkanCommandBuffer::VulkanCommandBuffer(VkDevice device,
                                         VkCommandBuffer commandBuffer,
                                         VkFence fence,
                                         VkQueue queue,
                                         VulkanContainers &containers)
    : device_(device), queue_(queue), commandBuffer_(commandBuffer),
      fence_(fence), containers_(containers) {}

VulkanCommandBuffer::~VulkanCommandBuffer() {
  wait();
}

void VulkanCommandBuffer::begin() {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = 0;

  vkBeginCommandBuffer(commandBuffer_, &beginInfo);
}

void VulkanCommandBuffer::end() {
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point endStart, descriptorSetsStart, descriptorSetsEnd;
  if (isProfilingEnabled()) {
    endStart = Clock::now();
    descriptorSetsStart = Clock::now();
  }

  // Create descriptor set layouts, pool, and allocate descriptor sets
  auto descriptorResult =
      createDescriptorSets(dispatches(), containers_.shaderContainer,
                           containers_.descriptorSetLayoutContainer,
                           containers_.descriptorContainer);
  auto descriptorSetLayoutHandles = std::move(descriptorResult.layoutHandles);
  descriptorsHandle_ = std::move(descriptorResult.descriptorsHandle);

  if (isProfilingEnabled()) {
    descriptorSetsEnd = Clock::now();
  }

  if (!descriptorSetLayoutHandles.empty() && descriptorsHandle_) {
    const auto &descriptorSets =
        containers_.descriptorContainer.getDescriptorSets(descriptorsHandle_);

    // Create compute pipelines from descriptor set layouts
    Clock::time_point pipelinesStart, pipelinesEnd;
    if (isProfilingEnabled()) {
      pipelinesStart = Clock::now();
    }
    auto pipelineResult = createComputePipelines(
        dispatches(), descriptorSetLayoutHandles, containers_.shaderContainer,
        containers_.descriptorSetLayoutContainer,
        containers_.pipelineLayoutContainer, containers_.pipelineContainer);
    pipelineHandles_ = std::move(pipelineResult.pipelineHandles);
    const auto &pipelineLayouts = pipelineResult.pipelineLayouts;
    if (isProfilingEnabled()) {
      pipelinesEnd = Clock::now();
    }

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
    descriptorWrites.reserve(totalBindings);

    size_t dispatchIndex = 0;
    for (const auto &dispatch : dispatches()) {
      if (dispatch.isBarrier() || dispatch.isBufferUpdate()) {
        continue; // Skip non-shader dispatches — no descriptors to update
      }

      const auto &reflection =
          containers_.shaderContainer.getReflection(dispatch.shader());

      // Process each binding in the dispatch
      for (const auto &binding : dispatch.bindings()) {
        if (!binding.isHandle()) {
          // Skip data bindings (push constants) for now
          continue;
        }

        // Find the corresponding binding info from shader reflection
        // Skip push constants as they don't use descriptor bindings
        const BindingInfo *bindingInfo = nullptr;
        for (const auto &info : reflection.bindings) {
          if (info.type == BindingType::PushConstant) {
            continue;
          }
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
            containers_.bufferContainer.getBuffer(binding.getHandle());

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = bufferStruct.buffer;
        bufferInfo.offset = bufferStruct.offset;
        bufferInfo.range = bufferStruct.size();
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
    Clock::time_point updateDescriptorsStart, updateDescriptorsEnd;
    if (isProfilingEnabled()) {
      updateDescriptorsStart = Clock::now();
    }
    if (!descriptorWrites.empty()) {
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(descriptorWrites.size()),
                             descriptorWrites.data(), 0, nullptr);
    }
    if (isProfilingEnabled()) {
      updateDescriptorsEnd = Clock::now();
    }

    // Pre-fetch VkPipeline values for command recording
    const auto &vkPipelines =
        containers_.pipelineContainer.getPipelines(pipelineHandles_);

    // GPU timestamp profiling: create query pool and collect labels
    if (isProfilingEnabled() && containers_.timestampPeriod > 0.0f) {
      queryCount_ = 0;
      dispatchLabels_.clear();
      for (const auto &d : dispatches()) {
        if (!d.isBarrier() && !d.isBufferUpdate()) {
          queryCount_++;
          dispatchLabels_.push_back(d.label());
        }
      }
      if (queryCount_ > 0) {
        VkQueryPoolCreateInfo queryPoolInfo{};
        queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolInfo.queryCount = queryCount_ * 2;
        VK_CHECK(
            vkCreateQueryPool(device_, &queryPoolInfo, nullptr, &queryPool_));
        vkCmdResetQueryPool(commandBuffer_, queryPool_, 0, queryCount_ * 2);
      }
    }

    // Record commands: bind pipelines, descriptor sets, push constants, and
    // dispatch
    Clock::time_point recordCommandsStart, recordCommandsEnd;
    if (isProfilingEnabled()) {
      recordCommandsStart = Clock::now();
    }

    // Track which buffer handles have been written by prior dispatches
    // so we only insert barriers when a dispatch reads from a written buffer.
    std::unordered_set<size_t> writtenHandles;
    bool needsBarrier = false;

    auto insertBarrierIfNeeded = [&]() {
      if (!needsBarrier)
        return;
      VkMemoryBarrier shaderBarrier{};
      shaderBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      shaderBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      shaderBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &shaderBarrier, 0, nullptr, 0, nullptr);
      needsBarrier = false;
      writtenHandles.clear();
    };

    dispatchIndex = 0;
    for (const auto &dispatch : dispatches()) {
      // Handle explicit barrier dispatches
      if (dispatch.isBarrier()) {
        VkMemoryBarrier computeBarrier{};
        computeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        computeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer_,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &computeBarrier, 0, nullptr, 0, nullptr);
        needsBarrier = false;
        writtenHandles.clear();
        continue; // Do NOT increment dispatchIndex
      }

      // Handle inline buffer update dispatches
      if (dispatch.isBufferUpdate()) {
        // Insert compute→transfer barrier if prior compute wrote to buffers
        if (needsBarrier) {
          insertBarrierIfNeeded();
        }

        const auto &targetBuffer = containers_.bufferContainer.getBuffer(
            dispatch.bufferUpdateTarget());
        const auto &data = dispatch.bufferUpdateData();
        vkCmdUpdateBuffer(commandBuffer_, targetBuffer.buffer,
                          targetBuffer.offset,
                          static_cast<VkDeviceSize>(data.size()), data.data());

        // Transfer→compute barrier so subsequent shaders see the update
        VkMemoryBarrier transferBarrier{};
        transferBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        transferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        transferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &transferBarrier, 0, nullptr, 0, nullptr);
        needsBarrier = false;
        writtenHandles.clear();
        continue; // Do NOT increment dispatchIndex
      }

      // Check if this dispatch reads from any previously written buffer.
      // If so, insert a barrier before this dispatch.
      if (!writtenHandles.empty()) {
        for (const auto &binding : dispatch.bindings()) {
          if (binding.isHandle()) {
            if (writtenHandles.count(binding.getHandle().id())) {
              insertBarrierIfNeeded();
              break;
            }
          }
        }
      }

      // Bind the compute pipeline
      vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                        vkPipelines[dispatchIndex]);

      // Bind the descriptor set
      vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelineLayouts[dispatchIndex], 0, 1,
                              &descriptorSets[dispatchIndex], 0, nullptr);

      uint32_t pcOffset = 0;
      std::array<uint8_t, 128> pcData;

      // Push constants from data/scalar bindings
      for (const auto &binding : dispatch.bindings()) {
        if (binding.isScalar()) {
          uint32_t scalar = binding.getScalar<uint32_t>();
          memcpy(pcData.data() + pcOffset, &scalar, sizeof(uint32_t));
          pcOffset += sizeof(uint32_t);
        } else if (binding.isData()) {
          const auto &data = binding.getData();
          memcpy(pcData.data() + pcOffset, data.data(), data.size());
          pcOffset += static_cast<uint32_t>(data.size());
        }
      }
      if (pcOffset != 0) {
        vkCmdPushConstants(commandBuffer_, pipelineLayouts[dispatchIndex],
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, pcOffset,
                           pcData.data());
      }

      // Dispatch compute work using workgroup size
      // Scale down by dtypeVecSize and tgSize from shader reflection
      const auto &wgSize = dispatch.workgroupSize();
      const auto &reflection =
          containers_.shaderContainer.getReflection(dispatch.shader());
      const uint32_t dtypeVecSize = reflection.dtypeVecSize;
      const auto &tgSize = reflection.tgSize;

      const uint32_t scaleX = std::max(dtypeVecSize * tgSize.x, 1u);
      const uint32_t scaleY = std::max(tgSize.y, 1u);
      const uint32_t scaleZ = std::max(tgSize.z, 1u);

      const uint32_t dispatchX = (wgSize.x + scaleX - 1) / scaleX;
      const uint32_t dispatchY = (wgSize.y + scaleY - 1) / scaleY;
      const uint32_t dispatchZ = (wgSize.z + scaleZ - 1) / scaleZ;

      // GPU timestamp: write start
      if (queryPool_ != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(commandBuffer_,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool_,
                            dispatchIndex * 2);
      }

      vkCmdDispatch(commandBuffer_, dispatchX, dispatchY, dispatchZ);

      // GPU timestamp: write end
      if (queryPool_ != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(commandBuffer_,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool_,
                            dispatchIndex * 2 + 1);
      }

      // Track this dispatch's output handle as written.
      if (dispatch.outputHandle()) {
        writtenHandles.insert(dispatch.outputHandle().id());
        needsBarrier = true;
      } else {
        // Conservative fallback: any handle binding could be a write
        for (const auto &binding : dispatch.bindings()) {
          if (binding.isHandle()) {
            writtenHandles.insert(binding.getHandle().id());
          }
        }
        needsBarrier = true;
      }

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

    if (isProfilingEnabled()) {
      recordCommandsEnd = Clock::now();

      // Log profiling information
      auto toUs = [](auto duration) {
        return std::chrono::duration_cast<std::chrono::microseconds>(duration)
            .count();
      };
      logMsg("[VulkanCommandBuffer::end] createDescriptorSets: %lld us",
             toUs(descriptorSetsEnd - descriptorSetsStart));
      logMsg("[VulkanCommandBuffer::end] createComputePipelines: %lld us",
             toUs(pipelinesEnd - pipelinesStart));

      logMsg("[VulkanCommandBuffer::end] vkUpdateDescriptorSets: %lld us",
             toUs(updateDescriptorsEnd - updateDescriptorsStart));
      logMsg("[VulkanCommandBuffer::end] recordCommands: %lld us",
             toUs(recordCommandsEnd - recordCommandsStart));
    }
  }

  // End command buffer recording
  Clock::time_point endCommandBufferStart, endCommandBufferEnd;
  if (isProfilingEnabled()) {
    endCommandBufferStart = Clock::now();
  }
  VK_CHECK(vkEndCommandBuffer(commandBuffer_));

  if (isProfilingEnabled()) {
    endCommandBufferEnd = Clock::now();

    auto toUs = [](auto duration) {
      return std::chrono::duration_cast<std::chrono::microseconds>(duration)
          .count();
    };
    logMsg("[VulkanCommandBuffer::end] vkEndCommandBuffer: %lld us",
           toUs(endCommandBufferEnd - endCommandBufferStart));
    logMsg("[VulkanCommandBuffer::end] total: %lld us",
           toUs(endCommandBufferEnd - endStart));
  }
}

void VulkanCommandBuffer::submit() {
  // Reset fence to unsignaled state before submit
  VK_CHECK(vkResetFences(device_, 1, &fence_));

  // Submit to queue with fence
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer_;

  VK_CHECK(vkQueueSubmit(queue_, 1, &submitInfo, fence_));
  submitted_ = true;
}

void VulkanCommandBuffer::wait() {
  // If not submitted, nothing to wait for
  if (!submitted_) {
    return;
  }

  // Wait for the fence to be signaled (command buffer execution complete)
  VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
  submitted_ = false;

  // Read back GPU timestamp results and log per-dispatch timing
  if (queryPool_ != VK_NULL_HANDLE && queryCount_ > 0) {
    std::vector<uint64_t> timestamps(queryCount_ * 2);
    VkResult result = vkGetQueryPoolResults(
        device_, queryPool_, 0, queryCount_ * 2,
        timestamps.size() * sizeof(uint64_t), timestamps.data(),
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

    if (result == VK_SUCCESS) {
      const float periodNs = containers_.timestampPeriod;
      double totalUs = 0.0;

      // Aggregate by label for compact output
      std::unordered_map<std::string, double> labelTotals;
      std::unordered_map<std::string, int> labelCounts;
      for (uint32_t i = 0; i < queryCount_; ++i) {
        double durationUs =
            static_cast<double>(timestamps[i * 2 + 1] - timestamps[i * 2]) *
            periodNs / 1000.0;
        totalUs += durationUs;
        std::string label =
            (i < dispatchLabels_.size() && !dispatchLabels_[i].empty())
                ? dispatchLabels_[i]
                : "unknown";
        labelTotals[label] += durationUs;
        labelCounts[label]++;
      }

      // Sort by total time descending
      std::vector<std::pair<std::string, double>> sorted(labelTotals.begin(),
                                                         labelTotals.end());
      std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
      });
      for (const auto &[label, us] : sorted) {
        logMsg("[GPU Profile] %-30s %9.1f us  (%dx)", label.c_str(), us,
               labelCounts[label]);
      }
      logMsg("[GPU Profile] %-30s %9.1f us  (%d dispatches)", "TOTAL", totalUs,
             queryCount_);
    }

    vkDestroyQueryPool(device_, queryPool_, nullptr);
    queryPool_ = VK_NULL_HANDLE;
    queryCount_ = 0;
    dispatchLabels_.clear();
  }
}

void VulkanCommandBuffer::resubmit() {
  // Wait for any prior execution to complete
  wait();

  // Reset fence and re-submit the same recorded command buffer
  VK_CHECK(vkResetFences(device_, 1, &fence_));

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer_;

  VK_CHECK(vkQueueSubmit(queue_, 1, &submitInfo, fence_));
  submitted_ = true;
}

} // namespace cut
