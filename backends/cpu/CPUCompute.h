#pragma once

#include "CPUKernels.h"
#include "ThreadPool.h"

#include <ComputeInterface.h>

#include <memory>
#include <string_view>

namespace cut {

// Forward declarations
class CPUContainers;

/**
 * CPU buffer data structure.
 */
struct CPUBufferStruct {
  static constexpr std::string_view Name = "CPUBuffer";

  void *data = nullptr;
  size_t size = 0;
};

/**
 * CPU shader/kernel data structure.
 * Maps a kernel type to execute instead of SPIR-V.
 */
struct CPUShaderStruct {
  static constexpr std::string_view Name = "CPUShader";

  CPUKernelType kernelType;
};

/**
 * CPU implementation of ComputeInterface.
 * Executes compute operations on the CPU using a thread pool.
 */
class CPUCompute : public ComputeInterface {
public:
  /**
   * Constructs a CPUCompute instance.
   * @param numThreads Number of worker threads (0 = hardware_concurrency).
   */
  explicit CPUCompute(size_t numThreads = 0);

  ~CPUCompute() override;

  /**
   * Creates a CPU buffer with optional initial data.
   * @param size Buffer size in bytes.
   * @param srcPtr Optional pointer to source data for initialization.
   * @param immutable Ignored for CPU backend (all buffers are read/write).
   * @return Handle to the created buffer.
   */
  ComputeHandle createBuffer(size_t size,
                             const void *srcPtr = nullptr,
                             bool immutable = false) override;

  /**
   * Copies data from host memory to a CPU buffer.
   */
  void copyDataToBuffer(const void *srcPtr,
                        const ComputeHandle &dstBuffer,
                        size_t size,
                        size_t srcOffset,
                        size_t dstOffset,
                        bool useStaging = false,
                        bool wait = false) override;

  /**
   * Copies data from a CPU buffer to host memory.
   */
  void copyDataFromBuffer(const ComputeHandle &srcBuffer,
                          void *dstPtr,
                          size_t size,
                          size_t srcOffset,
                          size_t dstOffset,
                          bool useStaging = false,
                          bool wait = false) override;

  /**
   * Creates a shader module (maps kernel type from SPIR-V enum).
   * @param spirvCode Vector containing the SPIR-V bytecode (used for reflection
   * only).
   * @return Handle to the created shader module.
   */
  ComputeHandle
  createShaderModule(const std::vector<uint32_t> &spirvCode) override;

  /**
   * Creates a shader module from a kernel type directly.
   * @param kernelType The CPU kernel type to use.
   * @return Handle to the created shader module.
   */
  ComputeHandle createKernel(CPUKernelType kernelType);

  /**
   * Returns the number of worker threads in the thread pool.
   */
  size_t numThreads() const;

  /**
   * Get the thread pool for parallel execution.
   */
  ThreadPool &threadPool() { return *threadPool_; }

  /**
   * Get the containers.
   */
  CPUContainers &containers() { return *containers_; }

private:
  std::unique_ptr<CPUContainers> containers_;
  std::unique_ptr<ThreadPool> threadPool_;
};

} // namespace cut
