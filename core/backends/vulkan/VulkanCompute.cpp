#include "VulkanContainers.h"
#include <VulkanCommandBuffer.h>
#include <VulkanCompute.h>

#include <climits>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace cut {

constexpr size_t kAlignment = 16;

VulkanCompute::VulkanCompute(const std::shared_ptr<VulkanInstance> &instance,
                             VulkanContextConfig config)
    : instance_(instance) {
  const PhysicalDeviceAndQueueIndex physicalDeviceAndQueueIdx =
      pickPhysicalDevice(*instance_, config);

  const auto physicalDevice = physicalDeviceAndQueueIdx.physicalDevice;
  computeQueueFamilyIndex_ = physicalDeviceAndQueueIdx.queueIndex;
  const float queuePriority = 1.f;

  VkDeviceQueueCreateInfo queueCreateInfo = {};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex_;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkPhysicalDeviceFeatures deviceFeatures = {};
  deviceFeatures.shaderInt16 = VK_TRUE;

  // Enable 16-bit storage and shader float16 for Float16 support
  VkPhysicalDeviceShaderFloat16Int8Features featuresFloat16Int8 = {};
  featuresFloat16Int8.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  featuresFloat16Int8.shaderFloat16 = VK_TRUE;

  VkPhysicalDevice16BitStorageFeatures features16bit = {};
  features16bit.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  features16bit.pNext = &featuresFloat16Int8;
  features16bit.storageBuffer16BitAccess = VK_TRUE;
  features16bit.uniformAndStorageBuffer16BitAccess = VK_TRUE;

  // Build list of device extensions, checking support first
  const std::vector<const char *> wantedExtensions = {
      "VK_KHR_portability_subset", "VK_KHR_16bit_storage",
      "VK_KHR_shader_float16_int8", "VK_KHR_cooperative_matrix",
      "VK_KHR_shader_integer_dot_product"};

  uint32_t extCount = 0;
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount,
                                       nullptr);
  std::vector<VkExtensionProperties> availableExts(extCount);
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount,
                                       availableExts.data());

  // First pass: check which extensions are available (defer cooperative matrix)
  std::vector<const char *> deviceExtensions;
  bool hasCoopMatExt = false;
  for (const char *ext : wantedExtensions) {
    for (const auto &avail : availableExts) {
      if (strcmp(avail.extensionName, ext) == 0) {
        if (strcmp(ext, "VK_KHR_cooperative_matrix") == 0) {
          hasCoopMatExt = true; // Don't add yet — verify feature support first
        } else if (strcmp(ext, "VK_KHR_shader_integer_dot_product") == 0) {
          caps_.integerDotProduct = true;
          deviceExtensions.push_back(ext);
        } else {
          deviceExtensions.push_back(ext);
        }
        break;
      }
    }
  }

  // Enable cooperative matrix if extension + feature are both available
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures = {};
  coopMatFeatures.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  if (hasCoopMatExt) {
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatQuery = {};
    coopMatQuery.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &coopMatQuery;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    if (coopMatQuery.cooperativeMatrix) {
      coopMatFeatures.cooperativeMatrix = VK_TRUE;
      caps_.cooperativeMatrix = true;
      deviceExtensions.push_back("VK_KHR_cooperative_matrix");
      // Chain into feature request
      coopMatFeatures.pNext = featuresFloat16Int8.pNext;
      featuresFloat16Int8.pNext = &coopMatFeatures;
    }
  }

  VkDeviceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pNext = &features16bit;
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.queueCreateInfoCount = 1;
  createInfo.pEnabledFeatures = &deviceFeatures;
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

  VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device_));

  vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties_);
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties_);

  // Query subgroup size for cooperative matrix dispatch sizing
  VkPhysicalDeviceSubgroupProperties subgroupProps = {};
  subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 props2 = {};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &subgroupProps;
  vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
  caps_.backend = ComputeBackend::Vulkan;
  caps_.subgroupSize = subgroupProps.subgroupSize;

  logMsg("Vulkan device: %s (coopMat: %s, intDot: %s, subgroupSize: %u)",
         deviceProperties_.deviceName,
         caps_.cooperativeMatrix ? "yes" : "no",
         caps_.integerDotProduct ? "yes" : "no",
         caps_.subgroupSize);

#if CUT_USE_VMA
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_1;
  allocatorInfo.physicalDevice = physicalDevice;
  allocatorInfo.device = device_;
  allocatorInfo.instance = *owner_;
  allocatorInfo.preferredLargeHeapBlockSize = 16 * 1024 * 1024; // 16 MB

  VK_CHECK(vmaCreateAllocator(&allocatorInfo, &allocator_));
#endif

  // Create containers with device
  containers_ = std::make_unique<VulkanContainers>(device_);
  containers_->timestampPeriod = deviceProperties_.limits.timestampPeriod;

  // Timestamp valid-bit count for masking raw query results.
  {
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount,
                                             qfProps.data());
    if (computeQueueFamilyIndex_ < qfCount &&
        qfProps[computeQueueFamilyIndex_].timestampValidBits > 0) {
      containers_->timestampValidBits =
          qfProps[computeQueueFamilyIndex_].timestampValidBits;
    }
  }

  // Vma dependent initializations
  IF_VMA_ENABLED_THEN(containers_->bufferContainer.setAllocator(allocator_));

  // Create and set the command buffer container
  setCommandBufferContainer(std::make_unique<VulkanCommandBufferContainer>(
      device_, computeQueueFamilyIndex_, config.maxCommandBuffers,
      *containers_));

  // Create batched staging transfer manager
  staging_ = std::make_unique<VulkanStaging>(
      device_, computeQueueFamilyIndex_,
      [this](size_t s) { return createStagingBuffer(s); },
      [this](VulkanBufferStruct &b) { destroyStagingBuffer(b); });
}

PhysicalDeviceAndQueueIndex
VulkanCompute::pickPhysicalDevice(VkInstance instance,
                                  const VulkanContextConfig &config) {
  uint32_t physicalDeviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);

  if (physicalDeviceCount == 0) {
    logErr("No suitable GPU found.");
  }

  std::vector<VkPhysicalDevice> instPhysDevices(physicalDeviceCount);
  vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
                             instPhysDevices.data());

  // Explicit device index: config value, overridden by CUT_VULKAN_DEVICE.
  int requestedIndex = config.preferredDevice;
  if (const char *deviceEnv = std::getenv("CUT_VULKAN_DEVICE")) {
    requestedIndex = std::atoi(deviceEnv);
  }

  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  uint32_t computeQueueFamilyIndex = UINT_MAX;

  for (uint32_t devIdx = 0; devIdx < physicalDeviceCount; devIdx++) {
    const auto &device = instPhysDevices[devIdx];
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);

    // Skip CPU/software implementations (e.g. llvmpipe) — CUT targets real
    // GPUs only. Explicitly requesting one is an error.
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
      if (requestedIndex >= 0 && static_cast<int>(devIdx) == requestedIndex) {
        logErr("Requested Vulkan device %u is a CPU/software implementation "
               "(%s); CPU devices are not supported",
               devIdx, properties.deviceName);
      }
      continue;
    }

    // Find compute queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
      if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0)
        continue;

      // Explicit device index requested (config or CUT_VULKAN_DEVICE) —
      // skip default selection and only match the requested index.
      if (requestedIndex >= 0) {
        if (static_cast<int>(devIdx) == requestedIndex)
          return {device, i};
        break;
      }

      // Default selection: first compute device, prefer matching type
      if (physicalDevice == VK_NULL_HANDLE) {
        physicalDevice = device;
        computeQueueFamilyIndex = i;
      } else if (properties.deviceType == config.preferredType) {
        return {device, i};
      }
      break;
    }
  }

  if (physicalDevice == VK_NULL_HANDLE) {
    logErr("Physical device does not support compute queues.");
  }

  return {physicalDevice, computeQueueFamilyIndex};
}

void VulkanCompute::cleanup() {
  if (device_ != VK_NULL_HANDLE) {
    flushTransfers();
    vkDeviceWaitIdle(device_);
  }

  // Destroy staging transfer manager
  staging_.reset();

  // Destroy persistent transfer resources
  cleanupTransferResources();

  // Destroy command buffer container first — it holds ComputeHandle references
  // into the containers (pipelines, descriptors, etc.)
  setCommandBufferContainer({});

  // Drain the buffer cache before destroying containers
  if (containers_) {
    containers_->bufferContainer.drainCache();
  }

  // Now safe to destroy the containers
  containers_.reset();

  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
  }
}

VulkanCompute::~VulkanCompute() {
  cleanup();
}

void VulkanCompute::flushTransfers() {
  if (staging_)
    staging_->flush();
}

uint32_t
findMemoryType(uint32_t typeFilter,
               const VkPhysicalDeviceMemoryProperties &memoryProperties,
               VkMemoryPropertyFlags memoryPropertyFlag) {
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
        (memoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlag) ==
            memoryPropertyFlag) {
      return i;
    }
  }
  logErr("Failed to find suitable memory type");
}

ComputeHandle VulkanCompute::createBuffer(const std::vector<uint32_t> &shape,
                                          DataType dtype,
                                          const void *srcPtr,
                                          bool isUniform) {
  if (shape.empty()) {
    logErr("Cannot create buffer with empty shape");
  }

  const size_t alignedSize = ComputeBuffer::calculateAlignedSize(shape, dtype);

  // Default to device-only for optimal GPU performance
  constexpr bool deviceOnly = true;

  // Try to reuse a cached buffer of matching size (avoids Vulkan allocation)
  if (!isUniform) {
    auto cached = containers_->bufferContainer.tryAcquireCached(alignedSize);
    if (cached) {
      // Re-stamp the cached buffer with the requested shape and dtype
      cached->setDtype(dtype);
      cached->setShape(shape);
      auto handle = containers_->bufferContainer.create(std::move(*cached));
      if (srcPtr != nullptr) {
        const size_t actualSize =
            ComputeBuffer::calculateActualSize(shape, dtype);
        copyDataToBuffer(srcPtr, handle, actualSize, 0, 0, deviceOnly);
      }
      return handle;
    }
  }

  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = alignedSize;
  bufferInfo.usage = isUniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                               : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  // Add transfer flags for staging buffer support
  if (deviceOnly) {
    bufferInfo.usage |=
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }

  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VulkanBufferStruct bufferStruct;
  bufferStruct.setDtype(
      dtype); // Store element data type (must be set before setShape)
  bufferStruct.setShape(shape); // Store tensor shape and calculate aligned size

#if CUT_USE_VMA
  const auto &allocator = allocator_;

  const VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  const VmaAllocationCreateFlags allocationFlags =
      VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;

  VmaAllocationCreateInfo allocInfo = {};
  allocInfo.usage = memoryUsage;
  allocInfo.flags = allocationFlags;

  VK_CHECK(vmaCreateBuffer(allocator, &bufferInfo, &allocInfo,
                           &bufferStruct.buffer, &bufferStruct.allocation,
                           nullptr));
#else
  // Choose memory properties based on buffer type
  VkMemoryPropertyFlags memoryPropertyFlag;
  if (deviceOnly) {
    memoryPropertyFlag = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  } else {
    // Host-visible and coherent for direct CPU access
    memoryPropertyFlag = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }

  VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &bufferStruct.buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device_, bufferStruct.buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      memRequirements.memoryTypeBits, memoryProperties_, memoryPropertyFlag);

  if (vkAllocateMemory(device_, &allocInfo, nullptr, &bufferStruct.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(device_, bufferStruct.buffer, nullptr);
    logErr("Failed to allocate buffer memory (size=%zu MB, memoryType=%u, "
           "activeBuffers=%zu, activeBytes=%zu MB)",
           allocInfo.allocationSize / (1024 * 1024), allocInfo.memoryTypeIndex,
           containers_->bufferContainer.size(),
           activeBufferMemoryBytes() / (1024 * 1024));
  }

  VK_CHECK(
      vkBindBufferMemory(device_, bufferStruct.buffer, bufferStruct.memory, 0));

  // Only map host-visible memory
  if (memoryPropertyFlag & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    VK_CHECK(vkMapMemory(device_, bufferStruct.memory, 0, alignedSize, 0,
                         &bufferStruct.data));
  }
  bufferStruct.isCoherent =
      (memoryPropertyFlag & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
#endif

  auto handle = containers_->bufferContainer.create(std::move(bufferStruct));

  if (srcPtr != nullptr) {
    // Pass actualSize so copyDataToBuffer uses aligned copy
    const size_t actualSize = ComputeBuffer::calculateActualSize(shape, dtype);
    copyDataToBuffer(srcPtr, handle, actualSize, 0, 0, deviceOnly);
  }

  return handle;
}

ComputeHandle VulkanCompute::createBufferMapped(
    const std::vector<uint32_t> &shape,
    DataType dtype,
    const void *srcPtr,
    bool preferHost) {
  if (shape.empty()) {
    logErr("Cannot create buffer with empty shape");
  }

  const size_t alignedSize = ComputeBuffer::calculateAlignedSize(shape, dtype);

  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = alignedSize;
  bufferInfo.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VulkanBufferStruct bufferStruct;
  bufferStruct.setDtype(dtype);
  bufferStruct.setShape(shape);

#if CUT_USE_VMA
  VmaAllocationCreateInfo allocInfo = {};
  // preferHost: keep large buffers in system RAM instead of BAR/VRAM.
  allocInfo.usage =
      preferHost ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO;
  allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VmaAllocationInfo resultInfo = {};
  VK_CHECK(vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo,
                           &bufferStruct.buffer, &bufferStruct.allocation,
                           &resultInfo));
  bufferStruct.data = resultInfo.pMappedData;
  bufferStruct.isCoherent = true; // VMA mapped allocations are coherent
#else
  VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &bufferStruct.buffer));

  VkMemoryRequirements memReq;
  vkGetBufferMemoryRequirements(device_, bufferStruct.buffer, &memReq);

  VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReq.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memReq.memoryTypeBits, memoryProperties_, memFlags);
  if (preferHost) {
    // Prefer a host-visible type that is NOT device-local (BAR) so large
    // buffers stay in system RAM; fall back to the default selection.
    for (uint32_t i = 0; i < memoryProperties_.memoryTypeCount; ++i) {
      const VkMemoryPropertyFlags f =
          memoryProperties_.memoryTypes[i].propertyFlags;
      if ((memReq.memoryTypeBits & (1u << i)) && (f & memFlags) == memFlags &&
          (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
        allocInfo.memoryTypeIndex = i;
        break;
      }
    }
  }

  if (vkAllocateMemory(device_, &allocInfo, nullptr, &bufferStruct.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(device_, bufferStruct.buffer, nullptr);
    logErr("Failed to allocate host-visible buffer memory");
  }

  VK_CHECK(
      vkBindBufferMemory(device_, bufferStruct.buffer, bufferStruct.memory, 0));
  VK_CHECK(vkMapMemory(device_, bufferStruct.memory, 0, alignedSize, 0,
                       &bufferStruct.data));
  bufferStruct.isCoherent = true;
#endif

  auto handle = containers_->bufferContainer.create(std::move(bufferStruct));

  if (srcPtr != nullptr) {
    const size_t actualSize = ComputeBuffer::calculateActualSize(shape, dtype);
    copyDataToBuffer(srcPtr, handle, actualSize, 0, 0, false);
  }

  return handle;
}

ComputeHandle
VulkanCompute::createBufferView(const ComputeHandle &parent,
                                size_t byteOffset,
                                const std::vector<uint32_t> &shape,
                                DataType dtype) {
  const auto &parentBuffer = containers_->bufferContainer.getBuffer(parent);

  // Accumulate offset for nested views (view-of-a-view).
  // The parent might itself be a view with a non-zero base offset.
  const size_t totalOffset = parentBuffer.offset + byteOffset;

  // Validate final offset alignment against Vulkan device limits
  const VkDeviceSize minAlignment =
      std::max(static_cast<VkDeviceSize>(kAlignment),
               deviceProperties_.limits.minStorageBufferOffsetAlignment);
  if (totalOffset % minAlignment != 0) {
    logErr("Buffer view total offset %zu is not aligned to "
           "minStorageBufferOffsetAlignment (%llu)",
           totalOffset, static_cast<unsigned long long>(minAlignment));
  }

  const size_t viewSize = ComputeBuffer::calculateAlignedSize(shape, dtype);
  if (totalOffset + viewSize > parentBuffer.offset + parentBuffer.size()) {
    logErr("Buffer view (offset=%zu + size=%zu) exceeds parent buffer "
           "size (%zu from base offset %zu)",
           byteOffset, viewSize, parentBuffer.size(), parentBuffer.offset);
  }

  VulkanBufferStruct viewStruct;
  viewStruct.setDtype(dtype);
  viewStruct.setShape(shape);
  viewStruct.buffer = parentBuffer.buffer;
  viewStruct.offset = totalOffset;
  viewStruct.isCoherent = parentBuffer.isCoherent;
  viewStruct.isView_ = true;
  viewStruct.parentHandle_ = parent;

  if (parentBuffer.data != nullptr) {
    // parentBuffer.data already includes the parent's own offset,
    // so only add the child's relative byteOffset.
    viewStruct.data = static_cast<uint8_t *>(parentBuffer.data) + byteOffset;
  }

  return containers_->bufferContainer.create(std::move(viewStruct));
}

VulkanBufferStruct VulkanCompute::createStagingBuffer(size_t size) {
  // Align buffer size to 16 bytes
  const size_t alignedSize = (size + kAlignment - 1) & ~(kAlignment - 1);

  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = alignedSize;
  bufferInfo.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VulkanBufferStruct stagingBuffer;
  // Set shape as 1D buffer with size bytes (dtype defaults to Float32 which is
  // 4 bytes)
  stagingBuffer.setShape({static_cast<uint32_t>((size + 3) / 4)});

#if CUT_USE_VMA
  VmaAllocationCreateInfo allocInfo = {};
  allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
  allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VmaAllocationInfo allocationInfo;
  VK_CHECK(vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo,
                           &stagingBuffer.buffer, &stagingBuffer.allocation,
                           &allocationInfo));
  stagingBuffer.data = allocationInfo.pMappedData;
  stagingBuffer.isCoherent = true;
#else
  const VkMemoryPropertyFlags memoryPropertyFlag =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VK_CHECK(
      vkCreateBuffer(device_, &bufferInfo, nullptr, &stagingBuffer.buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device_, stagingBuffer.buffer,
                                &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      memRequirements.memoryTypeBits, memoryProperties_, memoryPropertyFlag);

  if (vkAllocateMemory(device_, &allocInfo, nullptr, &stagingBuffer.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(device_, stagingBuffer.buffer, nullptr);
    logErr("Failed to allocate staging buffer memory");
  }

  VK_CHECK(vkBindBufferMemory(device_, stagingBuffer.buffer,
                              stagingBuffer.memory, 0));

  VK_CHECK(vkMapMemory(device_, stagingBuffer.memory, 0, alignedSize, 0,
                       &stagingBuffer.data));
  stagingBuffer.isCoherent = true;
#endif

  return stagingBuffer;
}

void VulkanCompute::destroyStagingBuffer(VulkanBufferStruct &stagingBuffer) {
  if (stagingBuffer.buffer == VK_NULL_HANDLE) {
    return;
  }

#if CUT_USE_VMA
  vmaDestroyBuffer(allocator_, stagingBuffer.buffer, stagingBuffer.allocation);
#else
  if (stagingBuffer.data != nullptr) {
    vkUnmapMemory(device_, stagingBuffer.memory);
  }
  vkFreeMemory(device_, stagingBuffer.memory, nullptr);
  vkDestroyBuffer(device_, stagingBuffer.buffer, nullptr);
#endif

  stagingBuffer.buffer = VK_NULL_HANDLE;
  stagingBuffer.data = nullptr;
}

void VulkanCompute::ensureTransferResources() {
  if (transferCommandPool_ != VK_NULL_HANDLE)
    return;

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = computeQueueFamilyIndex_;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VK_CHECK(
      vkCreateCommandPool(device_, &poolInfo, nullptr, &transferCommandPool_));

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = transferCommandPool_;
  allocInfo.commandBufferCount = 1;
  VK_CHECK(
      vkAllocateCommandBuffers(device_, &allocInfo, &transferCommandBuffer_));

  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VK_CHECK(vkCreateFence(device_, &fenceInfo, nullptr, &transferFence_));

  vkGetDeviceQueue(device_, computeQueueFamilyIndex_, 0, &transferQueue_);
}

void VulkanCompute::cleanupTransferResources() {
  if (transferFence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_, transferFence_, nullptr);
    transferFence_ = VK_NULL_HANDLE;
  }
  if (transferCommandPool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, transferCommandPool_, nullptr);
    transferCommandPool_ = VK_NULL_HANDLE;
  }
  transferCommandBuffer_ = VK_NULL_HANDLE;
  transferQueue_ = VK_NULL_HANDLE;
}

void VulkanCompute::executeBufferCopy(VkBuffer srcBuffer,
                                      VkBuffer dstBuffer,
                                      VkDeviceSize size,
                                      VkDeviceSize srcOffset,
                                      VkDeviceSize dstOffset) {
  ensureTransferResources();

  // Reset and re-record the persistent command buffer
  VK_CHECK(vkResetCommandBuffer(transferCommandBuffer_, 0));

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  VK_CHECK(vkBeginCommandBuffer(transferCommandBuffer_, &beginInfo));

  VkBufferCopy copyRegion = {};
  copyRegion.srcOffset = srcOffset;
  copyRegion.dstOffset = dstOffset;
  copyRegion.size = size;

  vkCmdCopyBuffer(transferCommandBuffer_, srcBuffer, dstBuffer, 1, &copyRegion);

  VK_CHECK(vkEndCommandBuffer(transferCommandBuffer_));

  // Submit and wait using persistent fence
  VK_CHECK(vkResetFences(device_, 1, &transferFence_));

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &transferCommandBuffer_;

  VK_CHECK(vkQueueSubmit(transferQueue_, 1, &submitInfo, transferFence_));
  VK_CHECK(vkWaitForFences(device_, 1, &transferFence_, VK_TRUE, UINT64_MAX));
}

void VulkanCompute::copyDataToBuffer(const void *srcPtr,
                                     const ComputeHandle &dstBuffer,
                                     size_t size,
                                     size_t srcOffset,
                                     size_t dstOffset,
                                     bool useStaging,
                                     bool wait) {
  const auto &buffer = containers_->bufferContainer.getBuffer(dstBuffer);
  const bool localUseStaging = useStaging || (buffer.data == nullptr);

  if (buffer.size() < dstOffset + size) {
    logErr("Trying to write data outside destination buffer range.");
  }

  const size_t actualSize = buffer.calculateActualSize();
  const bool isFullCopy =
      srcOffset == 0 && dstOffset == 0 && size == actualSize;

  if (localUseStaging) {
    const size_t copySize = isFullCopy ? buffer.calculateAlignedSize() : size;

    void *stagingDst = staging_->reserve(copySize);
    copyActualToAligned(srcPtr, stagingDst, buffer, isFullCopy ? 0 : srcOffset,
                        0, size);
    staging_->recordCopy(buffer.buffer,
                         (isFullCopy ? 0 : dstOffset) + buffer.offset, copySize,
                         dstBuffer);
  } else {
    // Host-visible buffer - use unified copy function
    copyActualToAligned(srcPtr, buffer.data, buffer, srcOffset, dstOffset,
                        size);

    // Flush memory to make writes visible to GPU
    if (!buffer.isCoherent) {
      const size_t flushSize =
          isFullCopy ? buffer.calculateAlignedSize() : size;
#if CUT_USE_VMA
      if (buffer.isView_) {
        const auto &parentBuffer =
            containers_->bufferContainer.getBuffer(buffer.parentHandle_);
        vmaFlushAllocation(allocator_, parentBuffer.allocation,
                           dstOffset + buffer.offset, flushSize);
      } else {
        vmaFlushAllocation(allocator_, buffer.allocation, dstOffset, flushSize);
      }
#else
      VkMappedMemoryRange memoryRange = {};
      memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      memoryRange.memory = buffer.memory;
      memoryRange.offset = dstOffset + buffer.offset;
      memoryRange.size = flushSize;

      VK_CHECK(vkFlushMappedMemoryRanges(device_, 1, &memoryRange));
#endif
    }
  }
}

void VulkanCompute::copyDataFromBuffer(const ComputeHandle &srcBuffer,
                                       void *dstPtr,
                                       size_t size,
                                       size_t srcOffset,
                                       size_t dstOffset,
                                       bool useStaging,
                                       bool wait) {
  flushTransfers();
  const auto &buffer = containers_->bufferContainer.getBuffer(srcBuffer);
  const bool localUseStaging = useStaging || (buffer.data == nullptr);

  if (buffer.size() < srcOffset + size) {
    logErr("Trying to read data outside source buffer range.");
  }

  const size_t actualSize = buffer.calculateActualSize();
  const bool isFullCopy =
      srcOffset == 0 && dstOffset == 0 && size == actualSize;

  if (localUseStaging) {
    const size_t copySize = isFullCopy ? buffer.calculateAlignedSize() : size;
    VulkanBufferStruct stagingBuffer = createStagingBuffer(copySize);

    executeBufferCopy(buffer.buffer, stagingBuffer.buffer, copySize,
                      (isFullCopy ? 0 : srcOffset) + buffer.offset, 0);
    copyAlignedToActual(stagingBuffer.data, dstPtr, buffer, 0,
                        isFullCopy ? 0 : dstOffset, size);
    destroyStagingBuffer(stagingBuffer);
  } else {
    // Invalidate memory to make GPU writes visible to CPU
    if (!buffer.isCoherent) {
      const size_t invalidateSize =
          isFullCopy ? buffer.calculateAlignedSize() : size;
#if CUT_USE_VMA
      if (buffer.isView_) {
        const auto &parentBuffer =
            containers_->bufferContainer.getBuffer(buffer.parentHandle_);
        vmaInvalidateAllocation(allocator_, parentBuffer.allocation,
                                srcOffset + buffer.offset, invalidateSize);
      } else {
        vmaInvalidateAllocation(allocator_, buffer.allocation, srcOffset,
                                invalidateSize);
      }
#else
      VkMappedMemoryRange memoryRange = {};
      memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      memoryRange.memory = buffer.memory;
      memoryRange.offset = srcOffset + buffer.offset;
      memoryRange.size = invalidateSize;

      VK_CHECK(vkInvalidateMappedMemoryRanges(device_, 1, &memoryRange));
#endif
    }

    // Host-visible buffer - use unified copy function
    copyAlignedToActual(buffer.data, dstPtr, buffer, srcOffset, dstOffset,
                        size);
  }
}

ComputeHandle
VulkanCompute::createShaderModule(const std::vector<uint32_t> &spirvCode) {
  return containers_->shaderContainer.createShader(spirvCode);
}

const ComputeBuffer &
VulkanCompute::getBuffer(const ComputeHandle &bufferHandle) const {
  static_assert(std::is_base_of<ComputeBuffer, VulkanBufferStruct>::value,
                "VulkanBufferStruct must derive from ComputeBuffer");
  return containers_->bufferContainer.getBuffer(bufferHandle);
}

size_t VulkanCompute::bufferCount() const {
  return containers_->bufferContainer.size();
}

size_t VulkanCompute::activeBufferMemoryBytes() const {
  return containers_->bufferContainer.activeMemoryBytes();
}

size_t VulkanCompute::deviceTotalMemoryBytes() const {
  size_t total = 0;
  for (uint32_t i = 0; i < memoryProperties_.memoryHeapCount; ++i) {
    if (memoryProperties_.memoryHeaps[i].flags &
        VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total += memoryProperties_.memoryHeaps[i].size;
    }
  }
  return total;
}

size_t VulkanCompute::bufferOffsetAlignment() const {
  return std::max(static_cast<VkDeviceSize>(16),
                  deviceProperties_.limits.minStorageBufferOffsetAlignment);
}

void VulkanCompute::releaseLoadingResources() {
  // Drain cached (freed but not destroyed) buffers — these are intermediates
  // from transpose ops etc. that were cached for potential reuse.
  containers_->bufferContainer.drainCache();

  // Flush and release staging buffer memory. The staging manager remains
  // alive (lazy-reallocated on next use) but its large buffer is freed.
  if (staging_) {
    staging_->flush();
    staging_->releaseStagingMemory();
  }
}

// Debug callback for validation layer messages
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  const char *severity = "UNKNOWN";
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    severity = "ERROR";
  } else if (messageSeverity &
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    severity = "WARNING";
  } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    severity = "INFO";
  } else if (messageSeverity &
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    severity = "VERBOSE";
  }

  const char *type = "GENERAL";
  if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) {
    type = "VALIDATION";
  } else if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) {
    type = "PERFORMANCE";
  }

  fprintf(stderr, "[Vulkan %s] [%s]: %s\n", severity, type,
          pCallbackData->pMessage);

  return VK_FALSE;
}

VulkanInstance::VulkanInstance() {
  // Validation layers add significant overhead (2-10x on Vulkan API calls).
  // Enable via CUT_VULKAN_VALIDATION=1 environment variable when debugging.
  const bool validation = [] {
    const char *env = std::getenv("CUT_VULKAN_VALIDATION");
    return env && std::string(env) == "1";
  }();

  const char *validationLayerName = "VK_LAYER_KHRONOS_validation";

  // MoltenVK extensions
  std::vector<const char *> requestedExtensions = {
      VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
  };

  // Add debug utils extension for validation messages
  if (validation) {
    requestedExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  std::vector<std::string> supportedExtensions;

  { // Get supported extensions
    uint32_t extensionCount = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                                    nullptr));

    std::vector<VkExtensionProperties> extensionProperties(extensionCount);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(
        nullptr, &extensionCount, extensionProperties.data()));

    supportedExtensions.reserve(extensionProperties.size());
    for (auto &extensionProperty : extensionProperties) {
      supportedExtensions.emplace_back(extensionProperty.extensionName);
    }
    // logMsg("Supported extensions", supportedExtensions);
  }

  std::vector<const char *> extensions;

  // Enable requested instance extensions (skip unsupported ones)
  if (!requestedExtensions.empty()) {
    for (const auto &requestedExtension : requestedExtensions) {
      bool found = false;
      for (const auto &supportedExtension : supportedExtensions) {
        if (supportedExtension == requestedExtension) {
          found = true;
          break;
        }
      }
      if (found) {
        extensions.push_back(requestedExtension);
      } else {
        logMsg("Optional extension %s is not supported, skipping.",
               requestedExtension);
      }
    }
  }

  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "CUT Vulkan";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "CUT";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  // Required for MoltenVK
  createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  createInfo.pApplicationInfo = &appInfo;
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

  //    if (validation) {
  //        createInfo.enabledExtensionCount =
  //        (uint32_t)instanceExtensions.size();
  //        createInfo.ppEnabledExtensionNames = instanceExtensions.data();
  //    }

  if (validation) {
    // Check if this layer is available at instance level
    uint32_t instanceLayerCount;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr));

    std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerCount);

    VK_CHECK(vkEnumerateInstanceLayerProperties(
        &instanceLayerCount, instanceLayerProperties.data()));

    std::vector<const char *> layerNames;
    for (const auto &layer : instanceLayerProperties) {
      layerNames.push_back(layer.layerName);
    }

    // logMsg("Instance Layers", layerNames);

    bool validationLayerPresent = false;
    for (const auto &layer : instanceLayerProperties) {
      if (std::strcmp(layer.layerName, validationLayerName) == 0) {
        validationLayerPresent = true;
        break;
      }
    }

    if (validationLayerPresent) {
      createInfo.ppEnabledLayerNames = &validationLayerName;
      createInfo.enabledLayerCount = 1;
    } else {
      logMsg("Validation layer not available, validation is disabled.");
    }
  }

  VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));

  // Set up debug messenger for validation messages
  if (validation) {
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
    debugCreateInfo.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = debugCallback;
    debugCreateInfo.pUserData = nullptr;

    auto createDebugUtilsMessenger =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));

    if (createDebugUtilsMessenger != nullptr) {
      VK_CHECK(createDebugUtilsMessenger(instance_, &debugCreateInfo, nullptr,
                                         &debugMessenger_));
    }
  }
}

std::unique_ptr<VulkanCompute>
VulkanInstance::createInterface(VulkanContextConfig config) {
  return std::make_unique<VulkanCompute>(this->getShared(), config);
}

VulkanInstance::~VulkanInstance() {
  // Destroy debug messenger before instance
  if (debugMessenger_ != VK_NULL_HANDLE) {
    auto destroyDebugUtilsMessenger =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_,
                                  "vkDestroyDebugUtilsMessengerEXT"));
    if (destroyDebugUtilsMessenger != nullptr) {
      destroyDebugUtilsMessenger(instance_, debugMessenger_, nullptr);
    }
  }

  // Destroy instance
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
  }
}

} // namespace cut
