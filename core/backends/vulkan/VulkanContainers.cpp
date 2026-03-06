#include <VulkanCompute.h>
#include <VulkanContainers.h>

#include <ComputeCommon.h>

namespace cut {

VulkanCommandBufferContainer::VulkanCommandBufferContainer(
    VkDevice device,
    uint32_t queueFamilyIndex,
    uint32_t maxCommandBuffers,
    VulkanContainers &containers)
    : VulkanContainerBase(device), containers_(containers) {
  vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue_);

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = queueFamilyIndex;

  VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool_));

  // Pre-allocate command buffers
  commandBuffers_.resize(maxCommandBuffers);

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool_;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = maxCommandBuffers;

  VK_CHECK(
      vkAllocateCommandBuffers(device, &allocInfo, commandBuffers_.data()));

  // Pre-allocate fences
  fences_.resize(maxCommandBuffers);

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = 0; // Unsignaled state

  for (uint32_t i = 0; i < maxCommandBuffers; ++i) {
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fences_[i]));
  }
}

VulkanCommandBufferContainer::~VulkanCommandBufferContainer() {
  if (queue_ != VK_NULL_HANDLE) {
    vkQueueWaitIdle(queue_);
  }
  for (auto fence : fences_) {
    if (fence != VK_NULL_HANDLE) {
      vkDestroyFence(getDevice(), fence, nullptr);
    }
  }
  if (commandPool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(getDevice(), commandPool_, nullptr);
  }
}

ComputeHandle VulkanCommandBufferContainer::createCommandBuffer() {
  // Get the next command buffer and fence in round-robin fashion
  VkCommandBuffer commandBuffer = commandBuffers_[nextBufferIndex_];
  VkFence fence = fences_[nextBufferIndex_];
  nextBufferIndex_ = (nextBufferIndex_ + 1) % commandBuffers_.size();

  return ComputeDataContainer::create(new VulkanCommandBuffer(
      getDevice(), commandBuffer, fence, queue_, containers_));
}

void VulkanBufferContainer::destroyAPIObject(const ComputeHandle &handle) {
  auto &buffer = get(handle);

  // Views share the parent's VkBuffer — do not destroy GPU resources.
  // The parentHandle_ ref is released when the struct is reset to default.
  if (buffer.isView_) {
    return;
  }

  activeMemoryBytes_ -= buffer.size();

#if CUT_USE_VMA
  if (buffer.data != nullptr) {
    vmaUnmapMemory(allocator_, buffer.allocation);
  }

  if (buffer.buffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);
  }
#else
  if (buffer.data != nullptr) {
    vkUnmapMemory(getDevice(), buffer.memory);
  }

  if (buffer.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(getDevice(), buffer.buffer, nullptr);
  }

  if (buffer.memory != VK_NULL_HANDLE) {
    vkFreeMemory(getDevice(), buffer.memory, nullptr);
  }
#endif
}

ComputeHandle
VulkanShaderContainer::createShader(const std::vector<uint32_t> &spirvCode) {
  VulkanShaderStruct shaderStruct{};

  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
  createInfo.pCode = spirvCode.data();

  VK_CHECK(vkCreateShaderModule(getDevice(), &createInfo, nullptr,
                                &shaderStruct.shader));

  // Store reflection data for descriptor set layout creation
  shaderStruct.reflection = reflectSpirvBindings(spirvCode);

  return ComputeDataContainer::create(std::move(shaderStruct));
}

void VulkanShaderContainer::destroyAPIObject(const ComputeHandle &handle) {
  const auto &shaderData = get(handle);
  vkDestroyShaderModule(getDevice(), shaderData.shader, nullptr);
}

ComputeHandle VulkanDescriptorContainer::findCachedDescriptor(
    const std::vector<ComputeHandle> &descriptorSetLayoutHandles,
    const std::vector<VkDescriptorPoolSize> &poolSizes) {
  for (const auto &handle : descriptorCache_) {
    if (handle && getRefCount(handle) == 1) {
      const auto &cached = ComputeDataContainer::get(handle);
      if (cached.descriptorSetLayoutHandles == descriptorSetLayoutHandles &&
          cached.poolSizes.size() == poolSizes.size() &&
          std::memcmp(cached.poolSizes.data(), poolSizes.data(),
                      cached.poolSizes.size() * sizeof(VkDescriptorPoolSize)) ==
              0) {
        return handle;
      }
    }
  }
  return {};
}

ComputeHandle VulkanDescriptorContainer::createDescriptorSets(
    const std::vector<VkDescriptorPoolSize> &poolSizes,
    const std::vector<ComputeHandle> &descriptorSetLayoutHandles,
    const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts) {
  // Check if descriptor sets with these layouts already exist
  ComputeHandle cachedHandle =
      findCachedDescriptor(descriptorSetLayoutHandles, poolSizes);
  if (cachedHandle) {
    return cachedHandle;
  }

  const uint32_t maxSets =
      static_cast<uint32_t>(descriptorSetLayoutHandles.size());

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = maxSets;

  VulkanDescriptorStruct poolStruct{};
  VK_CHECK(vkCreateDescriptorPool(getDevice(), &poolInfo, nullptr,
                                  &poolStruct.pool));

  // Store pool sizes and layout handles for cache lookup
  poolStruct.poolSizes = poolSizes;
  poolStruct.descriptorSetLayoutHandles = descriptorSetLayoutHandles;

  // Allocate descriptor sets
  poolStruct.descriptorSets.resize(descriptorSetLayouts.size());

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = poolStruct.pool;
  allocInfo.descriptorSetCount =
      static_cast<uint32_t>(descriptorSetLayouts.size());
  allocInfo.pSetLayouts = descriptorSetLayouts.data();

  VK_CHECK(vkAllocateDescriptorSets(getDevice(), &allocInfo,
                                    poolStruct.descriptorSets.data()));

  auto handle = ComputeDataContainer::create(std::move(poolStruct));
  descriptorCache_.emplace_back(handle);
  return handle;
}

void VulkanDescriptorContainer::destroyAPIObject(const ComputeHandle &handle) {
  auto &poolStruct = get(handle);
  if (poolStruct.pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(getDevice(), poolStruct.pool, nullptr);
  }
}

namespace {

/// Compares two VkDescriptorSetLayoutBinding structs for equality.
bool bindingsEqual(const VkDescriptorSetLayoutBinding &a,
                   const VkDescriptorSetLayoutBinding &b) {
  return a.binding == b.binding && a.descriptorType == b.descriptorType &&
         a.descriptorCount == b.descriptorCount &&
         a.stageFlags == b.stageFlags &&
         a.pImmutableSamplers == b.pImmutableSamplers;
}

/// Compares two binding vectors for equality (assumes both are sorted).
bool bindingVectorsEqual(const std::vector<VkDescriptorSetLayoutBinding> &a,
                         const std::vector<VkDescriptorSetLayoutBinding> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (!bindingsEqual(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

} // namespace

ComputeHandle VulkanDescriptorSetLayoutContainer::findCachedLayout(
    const std::vector<VkDescriptorSetLayoutBinding> &bindings) {
  for (const auto &handle : layoutCache_) {
    if (handle) {
      const auto &cached = ComputeDataContainer::get(handle);
      if (bindingVectorsEqual(cached.bindings, bindings)) {
        return handle;
      }
    }
  }
  return {};
}

ComputeHandle VulkanDescriptorSetLayoutContainer::createLayout(
    const VkDescriptorSetLayoutCreateInfo &createInfo,
    const std::vector<VkDescriptorSetLayoutBinding> &bindings) {
  VulkanDescriptorSetLayoutStruct layoutStruct{};
  layoutStruct.bindings = bindings;
  VK_CHECK(vkCreateDescriptorSetLayout(getDevice(), &createInfo, nullptr,
                                       &layoutStruct.layout));
  auto handle = ComputeDataContainer::create(std::move(layoutStruct));
  layoutCache_.emplace_back(handle);
  return handle;
}

std::vector<ComputeHandle> VulkanDescriptorSetLayoutContainer::createLayouts(
    const std::vector<std::vector<VkDescriptorSetLayoutBinding>>
        &layoutBindings) {
  std::vector<ComputeHandle> handles;
  handles.reserve(layoutBindings.size());

  for (const auto &bindings : layoutBindings) {
    // Check if a layout with these bindings already exists
    ComputeHandle cachedHandle = findCachedLayout(bindings);
    if (cachedHandle) {
      handles.emplace_back(cachedHandle);
      continue;
    }

    // Create a new layout
    VkDescriptorSetLayoutCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    createInfo.pBindings = bindings.empty() ? nullptr : bindings.data();

    handles.emplace_back(createLayout(createInfo, bindings));
  }

  return handles;
}

std::vector<VkDescriptorSetLayout>
VulkanDescriptorSetLayoutContainer::getLayouts(
    const std::vector<ComputeHandle> &handles) const {
  std::vector<VkDescriptorSetLayout> layouts;
  layouts.reserve(handles.size());
  for (const auto &handle : handles) {
    layouts.emplace_back(getLayout(handle));
  }
  return layouts;
}

void VulkanDescriptorSetLayoutContainer::destroyAPIObject(
    const ComputeHandle &handle) {
  auto &layoutStruct = get(handle);
  if (layoutStruct.layout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(getDevice(), layoutStruct.layout, nullptr);
  }
}

ComputeHandle VulkanPipelineLayoutContainer::createLayout(
    const VulkanPipelineLayoutCreateInfo &createInfo) {
  VkDescriptorSetLayout dsLayout = (createInfo.createInfo.setLayoutCount > 0 &&
                                    createInfo.createInfo.pSetLayouts)
                                       ? createInfo.createInfo.pSetLayouts[0]
                                       : VK_NULL_HANDLE;
  uint32_t pcSize = createInfo.createInfo.pushConstantRangeCount > 0
                        ? createInfo.pushConstantRange.size
                        : 0;

  LayoutCacheKey key{dsLayout, pcSize};
  auto it = layoutCache_.find(key);
  if (it != layoutCache_.end()) {
    return it->second;
  }

  VulkanPipelineLayoutStruct layoutStruct{};
  VK_CHECK(vkCreatePipelineLayout(getDevice(), &createInfo.createInfo, nullptr,
                                  &layoutStruct.layout));
  auto handle = ComputeDataContainer::create(std::move(layoutStruct));
  layoutCache_.emplace(key, handle);
  return handle;
}

std::vector<ComputeHandle> VulkanPipelineLayoutContainer::createLayouts(
    const std::vector<VulkanPipelineLayoutCreateInfo> &createInfos) {
  std::vector<ComputeHandle> handles;
  handles.reserve(createInfos.size());
  for (const auto &createInfo : createInfos) {
    handles.emplace_back(createLayout(createInfo));
  }
  return handles;
}

std::vector<VkPipelineLayout> VulkanPipelineLayoutContainer::getLayouts(
    const std::vector<ComputeHandle> &handles) const {
  std::vector<VkPipelineLayout> layouts;
  layouts.reserve(handles.size());
  for (const auto &handle : handles) {
    layouts.emplace_back(getLayout(handle));
  }
  return layouts;
}

void VulkanPipelineLayoutContainer::destroyAPIObject(
    const ComputeHandle &handle) {
  auto &layoutStruct = get(handle);
  if (layoutStruct.layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(getDevice(), layoutStruct.layout, nullptr);
  }
}

VulkanPipelineContainer::VulkanPipelineContainer(VkDevice device)
    : VulkanContainerBase(device) {
  VkPipelineCacheCreateInfo cacheCreateInfo{};
  cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  VK_CHECK(vkCreatePipelineCache(device, &cacheCreateInfo, nullptr,
                                 &pipelineCache_));
}

VulkanPipelineContainer::~VulkanPipelineContainer() {
  if (pipelineCache_ != VK_NULL_HANDLE) {
    vkDestroyPipelineCache(getDevice(), pipelineCache_, nullptr);
  }
}

ComputeHandle VulkanPipelineContainer::findCachedPipeline(
    VkShaderModule shaderModule, const ComputeHandle &pipelineLayoutHandle) {
  for (const auto &handle : pipelineCache_handles_) {
    if (handle) {
      const auto &cached = ComputeDataContainer::get(handle);
      if (cached.shaderModule == shaderModule &&
          cached.pipelineLayoutHandle == pipelineLayoutHandle) {
        return handle;
      }
    }
  }
  return {};
}

std::vector<ComputeHandle> VulkanPipelineContainer::createPipelines(
    const std::vector<VkPipelineShaderStageCreateInfo> &shaderStages,
    const std::vector<VkPipelineLayout> &pipelineLayouts,
    const std::vector<ComputeHandle> &pipelineLayoutHandles) {
  if (shaderStages.empty()) {
    return {};
  }

  std::vector<ComputeHandle> handles;
  handles.reserve(shaderStages.size());

  // Track which pipelines need to be created (not cached)
  std::vector<size_t> indicesToCreate;
  std::vector<VkComputePipelineCreateInfo> createInfos;

  for (size_t i = 0; i < shaderStages.size(); ++i) {
    // Check if a pipeline with this shader and layout already exists
    ComputeHandle cachedHandle =
        findCachedPipeline(shaderStages[i].module, pipelineLayoutHandles[i]);
    if (cachedHandle) {
      handles.emplace_back(cachedHandle);
      continue;
    }

    // Need to create this pipeline
    handles.emplace_back(ComputeHandle{}); // Placeholder
    indicesToCreate.emplace_back(i);

    VkComputePipelineCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    createInfo.stage = shaderStages[i];
    createInfo.layout = pipelineLayouts[i];
    createInfos.emplace_back(createInfo);
  }

  // Create all non-cached pipelines in a single Vulkan call
  if (!createInfos.empty()) {
    std::vector<VkPipeline> vkPipelines(createInfos.size());
    VK_CHECK(vkCreateComputePipelines(
        getDevice(), pipelineCache_, static_cast<uint32_t>(createInfos.size()),
        createInfos.data(), nullptr, vkPipelines.data()));

    // Store each new pipeline in the container and update handles
    for (size_t j = 0; j < indicesToCreate.size(); ++j) {
      size_t i = indicesToCreate[j];
      VulkanPipelineStruct pipelineStruct{};
      pipelineStruct.computePipeline = vkPipelines[j];
      pipelineStruct.shaderModule = shaderStages[i].module;
      pipelineStruct.pipelineLayoutHandle = pipelineLayoutHandles[i];

      auto handle = ComputeDataContainer::create(std::move(pipelineStruct));
      pipelineCache_handles_.emplace_back(handle);
      handles[i] = handle;
    }
  }

  return handles;
}

std::vector<VkPipeline> VulkanPipelineContainer::getPipelines(
    const std::vector<ComputeHandle> &handles) const {
  std::vector<VkPipeline> pipelines;
  pipelines.reserve(handles.size());
  for (const auto &handle : handles) {
    pipelines.emplace_back(getPipeline(handle));
  }
  return pipelines;
}

void VulkanPipelineContainer::destroyAPIObject(const ComputeHandle &handle) {
  auto &pipelineStruct = get(handle);
  if (pipelineStruct.computePipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(getDevice(), pipelineStruct.computePipeline, nullptr);
  }
}

} // namespace cut
