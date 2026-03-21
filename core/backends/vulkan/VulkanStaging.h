#pragma once

#include <ComputeHandle.h>
#include <VulkanCommon.h>
#include <VulkanStructs.h>

#include <functional>
#include <vector>

namespace cut {

/// Batched host-to-device staging transfer manager.
///
/// Accumulates staging copies into a single command buffer submission instead
/// of one fence-wait per upload. Uses a single shared staging buffer with
/// suballocation to avoid per-upload vkAllocateMemory/vkFreeMemory overhead.
class VulkanStaging {
public:
  using CreateStagingFn = std::function<VulkanBufferStruct(size_t)>;
  using DestroyStagingFn = std::function<void(VulkanBufferStruct &)>;

  VulkanStaging(VkDevice device,
                uint32_t queueFamilyIndex,
                CreateStagingFn createFn,
                DestroyStagingFn destroyFn);
  ~VulkanStaging();

  VulkanStaging(const VulkanStaging &) = delete;
  VulkanStaging &operator=(const VulkanStaging &) = delete;

  /// Reserve space in the shared staging buffer.
  /// Returns a host-writable pointer for the caller to fill.
  /// May flush and reallocate if the buffer is full.
  void *reserve(size_t bytes);

  /// Record a copy from the most recent reserve() to a device buffer.
  /// Keeps dstHandle alive until flush() completes.
  void recordCopy(VkBuffer dstBuffer,
                  VkDeviceSize dstOffset,
                  VkDeviceSize size,
                  const ComputeHandle &dstHandle);

  /// Submit all pending copies, wait for completion, reset for next batch.
  void flush();

  /// Free the staging buffer memory. Next reserve() will reallocate.
  void releaseStagingMemory();

  bool hasPending() const { return recording_; }

private:
  static constexpr size_t kAlignment = 16;
  static constexpr size_t kMinCapacity = 16 * 1024 * 1024; // 16 MB

  CreateStagingFn createStagingFn_;
  DestroyStagingFn destroyStagingFn_;

  VkDevice device_;
  uint32_t queueFamilyIndex_;

  // Command recording
  VkCommandPool cmdPool_ = VK_NULL_HANDLE;
  VkCommandBuffer cmdBuf_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  bool recording_ = false;

  // Shared staging buffer with suballocation
  VulkanBufferStruct staging_;
  size_t capacity_ = 0;
  size_t offset_ = 0;

  // Prevent destination buffers from being freed before flush
  std::vector<ComputeHandle> dstHandles_;
};

} // namespace cut
