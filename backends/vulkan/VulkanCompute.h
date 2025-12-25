#pragma once

#include <ComputeInterface.h>
#include <VulkanContainers.h>
#include <VulkanStructs.h>

namespace cut {

class VulkanInstance;

/*
 * Class implementing Vulkan compute.
 */
class VulkanCompute : public ComputeInterface {
public:
  /// Destroys the VulkanCompute instance and releases all Vulkan resources.
  ~VulkanCompute();

  /// Creates a GPU buffer with tensor-like shape.
  /// The innermost dimension is rounded up to a multiple of 4 for alignment.
  ComputeHandle
  createBuffer(const std::vector<size_t> &shape, // Dimension-wise sizes
               DataType dtype,                   // Data type of each element
               const void *srcPtr = nullptr,     // Host source ptr
               bool isUniform = false) override; // Is uniform buffer

  /// Copies data from host memory to a GPU buffer.
  void copyDataToBuffer(
      const void *srcPtr,             // Host source ptr
      const ComputeHandle &dstBuffer, // Destination buffer handle
      size_t size,                    // Copy size
      size_t srcOffset,               // Source offset
      size_t dstOffset,               // Destination offset
      bool useStaging = false,        // Use staging buffer
      bool wait = false) override; // Wait for copy to finish before returning

  /// Copies data from a GPU buffer to host memory.
  void copyDataFromBuffer(
      const ComputeHandle &srcBuffer, // Source buffer handle
      void *dstPtr,                   // Host destination ptr
      size_t size,                    // Copy size
      size_t srcOffset,               // Source offset
      size_t dstOffset,               // Destination offset
      bool useStaging = false,        // Use staging buffer
      bool wait = false) override; // Wait for copy to finish before returning

  /// Creates a shader module from SPIR-V bytecode.
  ComputeHandle
  createShaderModule(const std::vector<uint32_t> &spirvCode) override;

  /// Constructs a VulkanCompute instance with the given Vulkan instance and
  /// configuration.
  VulkanCompute(const std::shared_ptr<VulkanInstance> &instance,
                VulkanContextConfig config = {});

private:
  /// Selects a physical device (GPU) that matches the preferred type and has a
  /// compute queue.
  PhysicalDeviceAndQueueIndex pickPhysicalDevice(VkInstance instance,
                                                 VkPhysicalDeviceType type);
  /// Releases all Vulkan resources held by this instance.
  void cleanup();

  /// Creates a GPU buffer with specified memory type.
  /// @param size Buffer size in bytes (aligned).
  /// @param deviceOnly If true, allocates device-local memory (requires staging
  /// for CPU access).
  /// @param srcPtr Optional host data to initialize the buffer.
  /// @param isUniform If true, creates a uniform buffer; otherwise storage.
  /// @param shape Tensor shape (required when srcPtr is not null).
  /// @param dtype Data type of elements (required when srcPtr is not null).
  /// @return Handle to the created buffer.
  ComputeHandle createBuffer(size_t size,
                             bool deviceOnly,
                             const void *srcPtr,
                             bool isUniform,
                             const std::vector<size_t> &shape,
                             DataType dtype);

  /// Creates a staging buffer for transfer operations.
  /// @param size Size of the staging buffer in bytes.
  /// @return VulkanBufferStruct for the staging buffer.
  VulkanBufferStruct createStagingBuffer(size_t size);

  /// Destroys a staging buffer and frees its resources.
  /// @param stagingBuffer The staging buffer to destroy.
  void destroyStagingBuffer(VulkanBufferStruct &stagingBuffer);

  /// Executes a buffer-to-buffer copy using a one-shot command buffer.
  /// @param srcBuffer Source Vulkan buffer.
  /// @param dstBuffer Destination Vulkan buffer.
  /// @param size Number of bytes to copy.
  /// @param srcOffset Offset in the source buffer.
  /// @param dstOffset Offset in the destination buffer.
  void executeBufferCopy(VkBuffer srcBuffer,
                         VkBuffer dstBuffer,
                         VkDeviceSize size,
                         VkDeviceSize srcOffset,
                         VkDeviceSize dstOffset);

private:
  std::shared_ptr<VulkanInstance> instance_;

  uint32_t computeQueueFamilyIndex_ = 0;
  VkDevice device_ = VK_NULL_HANDLE;

  VkPhysicalDeviceProperties deviceProperties_ = {};
  VkPhysicalDeviceMemoryProperties memoryProperties_ = {};

  // VMA dependent allocations
  IF_VMA_ENABLED_THEN(VmaAllocator allocator_ = VK_NULL_HANDLE);

  std::unique_ptr<VulkanContainers> containers_;
};

class VulkanInstance : public std::enable_shared_from_this<VulkanInstance> {
public:
  /// Creates and initializes a Vulkan instance.
  VulkanInstance();
  /// Destroys the Vulkan instance.
  ~VulkanInstance();

  /// Creates a VulkanCompute interface for GPU compute operations.
  std::unique_ptr<VulkanCompute>
  createInterface(VulkanContextConfig config = {});

  /// Implicit conversion to the underlying VkInstance handle.
  operator VkInstance() const { return instance_; };

private:
  /// Returns a shared pointer to this instance.
  std::shared_ptr<VulkanInstance> getShared() { return shared_from_this(); }

  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
  std::array<VkPhysicalDevice, 1> physicalDevices_ = {};
};

} // namespace cut
