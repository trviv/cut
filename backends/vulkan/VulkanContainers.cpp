#include <VulkanCompute.h>
#include <VulkanContainers.h>

namespace cut {

VulkanCommandBufferContainer::VulkanCommandBufferContainer(
    VkDevice device,
    uint32_t queueFamilyIndex,
    VulkanBufferContainer &bufferContainer,
    VulkanShaderContainer &shaderContainer,
    VulkanDescriptorContainer &descriptorContainer,
    VulkanDescriptorSetLayoutContainer &descriptorSetLayoutContainer,
    VulkanPipelineLayoutContainer &pipelineLayoutContainer)
    : bufferContainer_(bufferContainer), shaderContainer_(shaderContainer),
      descriptorContainer_(descriptorContainer),
      descriptorSetLayoutContainer_(descriptorSetLayoutContainer),
      pipelineLayoutContainer_(pipelineLayoutContainer) {
  setDevice(device);
  vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue_);

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = queueFamilyIndex;

  VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool_));
}

VulkanCommandBufferContainer::~VulkanCommandBufferContainer() {
  if (queue_ != VK_NULL_HANDLE) {
    vkQueueWaitIdle(queue_);
  }
  if (commandPool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(getDevice(), commandPool_, nullptr);
  }
}

void VulkanBufferContainer::destroyAPIObject(const ComputeHandle &handle) {
  auto &buffer = get(handle);

#if CUT_USE_VMA
  if (buffer.mappedData != nullptr) {
    vmaUnmapMemory(allocator_, buffer.allocation);
  }

  if (buffer.buffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);
  }
#else
  if (buffer.mappedData != nullptr) {
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

void VulkanShaderContainer::destroyAPIObject(const ComputeHandle &handle) {
  const auto &shaderData = get(handle);
  vkDestroyShaderModule(getDevice(), shaderData.shader, nullptr);
}

ComputeHandle VulkanDescriptorContainer::createDescriptorSets(
    const std::vector<VkDescriptorPoolSize> &poolSizes,
    const std::vector<ComputeHandle> &descriptorSetLayoutHandles,
    const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts) {
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

  // Store layout handles for reference
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

  return ComputeDataContainer::create(std::move(poolStruct));
}

void VulkanDescriptorContainer::destroyAPIObject(const ComputeHandle &handle) {
  auto &poolStruct = get(handle);
  if (poolStruct.pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(getDevice(), poolStruct.pool, nullptr);
  }
}

ComputeHandle VulkanDescriptorSetLayoutContainer::createLayout(
    const VkDescriptorSetLayoutCreateInfo &createInfo) {
  VulkanDescriptorSetLayoutStruct layoutStruct{};
  VK_CHECK(vkCreateDescriptorSetLayout(getDevice(), &createInfo, nullptr,
                                       &layoutStruct.layout));
  return ComputeDataContainer::create(std::move(layoutStruct));
}

std::vector<ComputeHandle> VulkanDescriptorSetLayoutContainer::createLayouts(
    const VkDescriptorSetLayoutCreateInfo *createInfos, size_t count) {
  std::vector<ComputeHandle> handles;
  handles.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    handles.emplace_back(createLayout(createInfos[i]));
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
  VulkanPipelineLayoutStruct layoutStruct{};
  VK_CHECK(vkCreatePipelineLayout(getDevice(), &createInfo.createInfo, nullptr,
                                  &layoutStruct.layout));
  return ComputeDataContainer::create(std::move(layoutStruct));
}

std::vector<ComputeHandle> VulkanPipelineLayoutContainer::createLayouts(
    const VulkanPipelineLayoutCreateInfo *createInfos, size_t count) {
  std::vector<ComputeHandle> handles;
  handles.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    handles.emplace_back(createLayout(createInfos[i]));
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

std::vector<ComputeHandle> VulkanPipelineContainer::createPipelines(
    const std::vector<VkPipelineShaderStageCreateInfo> &shaderStages,
    const std::vector<VkPipelineLayout> &pipelineLayouts,
    const std::vector<ComputeHandle> &pipelineLayoutHandles) {
  if (shaderStages.empty()) {
    return {};
  }

  // Build compute pipeline create infos
  std::vector<VkComputePipelineCreateInfo> createInfos;
  createInfos.reserve(shaderStages.size());

  for (size_t i = 0; i < shaderStages.size(); ++i) {
    VkComputePipelineCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    createInfo.stage = shaderStages[i];
    createInfo.layout = pipelineLayouts[i];
    createInfos.emplace_back(createInfo);
  }

  // Create all pipelines in a single Vulkan call
  std::vector<VkPipeline> vkPipelines(createInfos.size());
  VK_CHECK(vkCreateComputePipelines(
      getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(createInfos.size()),
      createInfos.data(), nullptr, vkPipelines.data()));

  // Store each pipeline in the container
  std::vector<ComputeHandle> handles;
  handles.reserve(vkPipelines.size());

  for (size_t i = 0; i < vkPipelines.size(); ++i) {
    VulkanPipelineStruct pipelineStruct{};
    pipelineStruct.computePipeline = vkPipelines[i];
    pipelineStruct.pipelineLayoutHandle = pipelineLayoutHandles[i];
    handles.emplace_back(
        ComputeDataContainer::create(std::move(pipelineStruct)));
  }

  return handles;
}

std::vector<VkPipeline> VulkanComputePipelineContainer::getPipelines(
    const std::vector<ComputeHandle> &handles) const {
  std::vector<VkPipeline> pipelines;
  pipelines.reserve(handles.size());
  for (const auto &handle : handles) {
    pipelines.emplace_back(getPipeline(handle));
  }
  return pipelines;
}

void VulkanComputePipelineContainer::destroyAPIObject(
    const ComputeHandle &handle) {
  auto &pipelineStruct = get(handle);
  if (pipelineStruct.computePipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(getDevice(), pipelineStruct.computePipeline, nullptr);
  }
}

} // namespace cut
