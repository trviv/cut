#pragma once

#include "CPUContainers.h"
#include "CPUKernels.h"
#include "ThreadPool.h"

#include <ComputeStructs.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace cut {

// Forward declaration
class CPUCompute;

/**
 * Shared synchronization state for CPU command buffer tasks.
 * This is shared between the command buffer and its submitted tasks
 * to ensure safe access even if the command buffer is destroyed
 * before all tasks complete their notification.
 */
struct CPUCommandBufferSync {
  std::atomic<size_t> pendingTasks{0};
  std::mutex mutex;
  std::condition_variable cv;
};

/**
 * CPU implementation of CommandBuffer.
 * Executes compute dispatches on the CPU using a thread pool.
 */
class CPUCommandBuffer final : public CommandBuffer {
public:
  static constexpr std::string_view Name = "CPUCommandBuffer";

  CPUCommandBuffer(CPUContainers &containers,
                   ThreadPool &threadPool,
                   CPUCompute *compute);
  ~CPUCommandBuffer() override;

  void submit() override;
  void wait() override;

private:
  CPUContainers &containers_;
  ThreadPool &threadPool_;
  CPUCompute *compute_;
  std::shared_ptr<CPUCommandBufferSync> sync_;
  bool submitted_{false};
};

} // namespace cut
