#pragma once

#include "CPUContainers.h"
#include "ThreadPool.h"

#include <ComputeStructs.h>

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace cut {

/**
 * CPU implementation of CommandBuffer.
 * Executes compute dispatches on the CPU using a thread pool.
 */
class CPUCommandBuffer final : public CommandBuffer {
public:
  static constexpr std::string_view Name = "CPUCommandBuffer";

  /**
   * Constructs a CPUCommandBuffer with the necessary resources.
   * @param containers Reference to the CPU containers struct.
   * @param threadPool Reference to the thread pool for parallel execution.
   */
  CPUCommandBuffer(CPUContainers &containers, ThreadPool &threadPool);

  ~CPUCommandBuffer() override;

  /**
   * Submits the command buffer for execution on the CPU.
   * Dispatches workgroups to the thread pool for parallel execution.
   */
  void submit() override;

  /**
   * Waits for the command buffer to finish execution.
   * Blocks until all workgroups have completed.
   */
  void wait() override;

private:
  CPUContainers &containers_;
  ThreadPool &threadPool_;
  std::atomic<size_t> pendingWorkgroups_{0};
  std::mutex mutex_;
  std::condition_variable cv_;
  bool submitted_{false};
};

} // namespace cut
