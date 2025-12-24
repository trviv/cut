#pragma once

#include "CPUContainers.h"
#include "CPUStructs.h"
#include "ThreadPool.h"

#include <ComputeInterface.h>

#include <memory>

namespace cut {

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

  /// Destructor.
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
   * @param srcPtr Pointer to the source data.
   * @param dstBuffer Handle to the destination buffer.
   * @param size Number of bytes to copy.
   * @param srcOffset Offset in the source data.
   * @param dstOffset Offset in the destination buffer.
   * @param useStaging Ignored for CPU backend.
   * @param wait Ignored for CPU backend (always synchronous).
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
   * @param srcBuffer Handle to the source buffer.
   * @param dstPtr Pointer to the destination in host memory.
   * @param size Number of bytes to copy.
   * @param srcOffset Offset in the source buffer.
   * @param dstOffset Offset in the destination data.
   * @param useStaging Ignored for CPU backend.
   * @param wait Ignored for CPU backend (always synchronous).
   */
  void copyDataFromBuffer(const ComputeHandle &srcBuffer,
                          void *dstPtr,
                          size_t size,
                          size_t srcOffset,
                          size_t dstOffset,
                          bool useStaging = false,
                          bool wait = false) override;

  /**
   * Creates a shader module from SPIR-V bytecode.
   * Performs reflection to extract binding information.
   * The kernel function must be registered separately via registerKernel().
   * @param spirvCode Vector containing the SPIR-V bytecode.
   * @return Handle to the created shader module.
   */
  ComputeHandle
  createShaderModule(const std::vector<uint32_t> &spirvCode) override;

  /**
   * Registers a C++ kernel function for a shader handle.
   * Must be called before dispatching with this shader.
   * @param shaderHandle Handle to the shader.
   * @param kernel The C++ kernel function matching the GLSL shader logic.
   */
  void registerKernel(const ComputeHandle &shaderHandle, CPUKernel kernel);

  /**
   * Returns the number of worker threads in the thread pool.
   */
  size_t numThreads() const { return threadPool_->numThreads(); }

private:
  std::unique_ptr<CPUContainers> containers_;
  std::unique_ptr<ThreadPool> threadPool_;
};

} // namespace cut
