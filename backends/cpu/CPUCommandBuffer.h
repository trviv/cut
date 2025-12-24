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

  CPUCommandBuffer(CPUContainers &containers, ThreadPool &threadPool);
  ~CPUCommandBuffer() override;

  void submit() override;
  void wait() override;

private:
  CPUContainers &containers_;
  ThreadPool &threadPool_;
  std::atomic<size_t> pendingTasks_{0};
  std::mutex mutex_;
  std::condition_variable cv_;
  bool submitted_{false};
};

} // namespace cut
