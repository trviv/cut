#include "VulkanContainers.h"
#include <VulkanCommandBuffer.h>
#include <VulkanCompute.h>

#include <type_traits>

namespace cut {

constexpr size_t kAlignment = 16;

VulkanCompute::VulkanCompute(const std::shared_ptr<VulkanInstance> &instance,
                             VulkanContextConfig config)
    : instance_(instance) {
  const PhysicalDeviceAndQueueIndex physicalDeviceAndQueueIdx =
      pickPhysicalDevice(*instance_, config.preferredType);

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

  // Required extensions for MoltenVK
  const std::vector<const char *> deviceExtensions = {
      "VK_KHR_portability_subset", "VK_KHR_16bit_storage",
      "VK_KHR_shader_float16_int8"};

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

#if CUT_USE_VMA
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_1;
  allocatorInfo.physicalDevice = physicalDevice;
  allocatorInfo.device = device_;
  allocatorInfo.instance = *owner_;

  VK_CHECK(vmaCreateAllocator(&allocatorInfo, &allocator_));
#endif

  // Create containers with device
  containers_ = std::make_unique<VulkanContainers>(device_);

  // Vma dependent initializations
  IF_VMA_ENABLED_THEN(containers_->bufferContainer.setAllocator(allocator_));

  // Create and set the command buffer container
  setCommandBufferContainer(std::make_unique<VulkanCommandBufferContainer>(
      device_, computeQueueFamilyIndex_, config.maxCommandBuffers,
      *containers_));
}

PhysicalDeviceAndQueueIndex
VulkanCompute::pickPhysicalDevice(VkInstance instance,
                                  VkPhysicalDeviceType type) {
  uint32_t physicalDeviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);

  if (physicalDeviceCount == 0) {
    logErr("No suitable GPU found.");
  }

  std::vector<VkPhysicalDevice> instPhysDevices(physicalDeviceCount);
  vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
                             instPhysDevices.data());

  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  uint32_t computeQueueFamilyIndex = UINT_MAX;

  for (const auto &device : instPhysDevices) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);

    // Find compute queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
      // Only proceed if device supports compute
      if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
        continue;
      }

      // Use the first compute device, if no other have been found yet
      if (physicalDevice == VK_NULL_HANDLE) {
        physicalDevice = device;
        computeQueueFamilyIndex = i;
      } else if (properties.deviceType == type) {
        return {device, i};
      }
    }
  }

  if (physicalDevice == VK_NULL_HANDLE) {
    logErr("Physical device does not support compute queues.");
  }

  return {physicalDevice, computeQueueFamilyIndex};
}

void VulkanCompute::cleanup() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
  }

  // Destroy command buffer container first — it holds ComputeHandle references
  // into the containers (pipelines, descriptors, etc.)
  setCommandBufferContainer({});

  // Now safe to destroy the containers
  containers_.reset();

  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
  }
}

VulkanCompute::~VulkanCompute() {
  cleanup();
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
    logErr("Failed to allocate buffer memory");
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

ComputeHandle
VulkanCompute::createBufferView(const ComputeHandle &parent,
                                size_t byteOffset,
                                const std::vector<uint32_t> &shape,
                                DataType dtype) {
  const auto &parentBuffer = containers_->bufferContainer.getBuffer(parent);

  // Validate offset alignment against Vulkan device limits
  const VkDeviceSize minAlignment =
      std::max(static_cast<VkDeviceSize>(kAlignment),
               deviceProperties_.limits.minStorageBufferOffsetAlignment);
  if (byteOffset % minAlignment != 0) {
    logErr("Buffer view offset %zu is not aligned to "
           "minStorageBufferOffsetAlignment (%llu)",
           byteOffset, static_cast<unsigned long long>(minAlignment));
  }

  const size_t viewSize = ComputeBuffer::calculateAlignedSize(shape, dtype);
  if (byteOffset + viewSize > parentBuffer.size()) {
    logErr("Buffer view (offset=%zu + size=%zu) exceeds parent buffer "
           "size (%zu)",
           byteOffset, viewSize, parentBuffer.size());
  }

  VulkanBufferStruct viewStruct;
  viewStruct.setDtype(dtype);
  viewStruct.setShape(shape);
  viewStruct.buffer = parentBuffer.buffer;
  viewStruct.offset = byteOffset;
  viewStruct.isCoherent = parentBuffer.isCoherent;
  viewStruct.isView_ = true;
  viewStruct.parentHandle_ = parent;

  if (parentBuffer.data != nullptr) {
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

void VulkanCompute::executeBufferCopy(VkBuffer srcBuffer,
                                      VkBuffer dstBuffer,
                                      VkDeviceSize size,
                                      VkDeviceSize srcOffset,
                                      VkDeviceSize dstOffset) {
  // Create a one-shot command buffer for the copy operation
  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = computeQueueFamilyIndex_;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

  VkCommandPool commandPool;
  VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool));

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  VK_CHECK(vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer));

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

  VkBufferCopy copyRegion = {};
  copyRegion.srcOffset = srcOffset;
  copyRegion.dstOffset = dstOffset;
  copyRegion.size = size;

  vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

  VK_CHECK(vkEndCommandBuffer(commandBuffer));

  // Submit and wait for completion
  VkQueue queue;
  vkGetDeviceQueue(device_, computeQueueFamilyIndex_, 0, &queue);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

  VkFence fence;
  VK_CHECK(vkCreateFence(device_, &fenceInfo, nullptr, &fence));

  VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
  VK_CHECK(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX));

  // Cleanup
  vkDestroyFence(device_, fence, nullptr);
  vkDestroyCommandPool(device_, commandPool, nullptr);
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
    VulkanBufferStruct stagingBuffer = createStagingBuffer(copySize);

    copyActualToAligned(srcPtr, stagingBuffer.data, buffer,
                        isFullCopy ? 0 : srcOffset, 0, size);
    executeBufferCopy(stagingBuffer.buffer, buffer.buffer, copySize, 0,
                      (isFullCopy ? 0 : dstOffset) + buffer.offset);
    destroyStagingBuffer(stagingBuffer);
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

size_t VulkanCompute::bufferOffsetAlignment() const {
  return std::max(static_cast<VkDeviceSize>(16),
                  deviceProperties_.limits.minStorageBufferOffsetAlignment);
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
  const bool validation = true;

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
