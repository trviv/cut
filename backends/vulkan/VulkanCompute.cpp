#include "VulkanContainers.h"
#include <VulkanCommandBuffer.h>
#include <VulkanCompute.h>

namespace cut {

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

  // Required extensions for MoltenVK
  const std::vector<const char *> deviceExtensions = {
      "VK_KHR_portability_subset"};

  VkDeviceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
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
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_0;
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
      device_, computeQueueFamilyIndex_, *containers_));
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
  containers_.reset();

  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
  }

  // Delete / reset the container before destroying device
  setCommandBufferContainer({});

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

ComputeHandle
VulkanCompute::createBuffer(size_t size, const void *srcPtr, bool isUniform) {
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = isUniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                               : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VulkanBufferStruct bufferStruct;
  bufferStruct.size = size;

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
  // Note: Avoid HOST_CACHED_BIT as it can cause issues with GPU writes
  // on some platforms (especially with MoltenVK)
  const VkMemoryPropertyFlags memoryPropertyFlag =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  //        (isUniform ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
  //                   : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

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

  if (memoryPropertyFlag & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    VK_CHECK(vkMapMemory(device_, bufferStruct.memory, 0, size, 0,
                         &bufferStruct.mappedData));
  }
  if (memoryPropertyFlag & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
    bufferStruct.isCoherent = true;
  }
#endif

  auto handle = containers_->bufferContainer.create(std::move(bufferStruct));

  if (srcPtr != nullptr) {
    copyDataToBuffer(srcPtr, handle, size, 0, 0);
  }

  return handle;
}

void VulkanCompute::copyDataToBuffer(const void *srcPtr,
                                     const ComputeHandle &dstBuffer,
                                     size_t size,
                                     size_t srcOffset,
                                     size_t dstOffset,
                                     bool useStaging,
                                     bool wait) {
  const auto &buffer = containers_->bufferContainer.getBuffer(dstBuffer);
  const bool localUseStaging = useStaging || (buffer.mappedData == nullptr);

  if (buffer.size < dstOffset + size) {
    logErr("Trying to write data outside destination buffer range.");
  }

  if (localUseStaging) {
    logErr("Staging buffer copy is not yet implemented.");
  } else {
    if (!buffer.isCoherent) {
      logErr("Incoherent memory behaviour is not yet implemented.");
    }
    memcpy(buffer.mappedData, srcPtr, size);
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
  const bool localUseStaging = useStaging || (buffer.mappedData == nullptr);

  if (buffer.size < srcOffset + size) {
    logErr("Trying to read data outside source buffer range.");
  }

  if (localUseStaging) {
    logErr("Staging buffer copy is not yet implemented.");
  } else {
    if (!buffer.isCoherent) {
      logErr("Incoherent memory behaviour is not yet verified.");
#if CUT_USE_VMA
      vmaInvalidateAllocation(allocator_, buffer.allocation, srcOffset, size);
#else
      VkMappedMemoryRange memoryRanges = {};
      memoryRanges.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      memoryRanges.memory = buffer.memory;
      memoryRanges.offset = srcOffset;
      memoryRanges.size = size;

      VK_CHECK(vkInvalidateMappedMemoryRanges(device_, 1, &memoryRanges));
#endif
    }
    memcpy(dstPtr, buffer.mappedData, size);
  }
}

ComputeHandle
VulkanCompute::createShaderModule(const std::vector<uint32_t> &spirvCode) {
  return containers_->shaderContainer.createShader(spirvCode);
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
    logMsg("Supported extensions", supportedExtensions);
  }

  std::vector<const char *> extensions;

  // Enabled requested instance extensions
  if (!requestedExtensions.empty()) {
    for (const auto &requestedExtension : requestedExtensions) {
      bool found = false;
      // Output message if requested extension is not available
      for (const auto &supportedExtension : supportedExtensions) {
        if (supportedExtension == requestedExtension) {
          found = true;
          break;
        }
      }
      if (!found) {
        logErr("Requested extension %s is not supported.", requestedExtension);
      }
      extensions.push_back(requestedExtension);
    }
  }

  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "CUT Vulkan";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "CUT";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

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

    logMsg("Instance Layers", layerNames);

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
