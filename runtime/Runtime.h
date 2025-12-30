#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeInterface.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

// For SIMDMode enum
#include "../backends/cpu/CPUKernels.h"

#include <map>
#include <memory>
#include <vector>

namespace cut {

// Forward declarations
class VulkanInstance;
class VulkanCompute;
class CPUCompute;
class Dispatcher;

/**
 * Backend type enum for runtime selection.
 */
enum class BackendType { Vulkan, CPU };

/**
 * Runtime class that manages compute backend lifecycle and operator execution.
 * Provides a unified interface for executing compute operations across
 * different backends (Vulkan GPU, CPU with SIMD).
 */
class Runtime {
public:
  /**
   * Constructs a Runtime instance.
   * Does not initialize any backend - call init() to set up the compute
   * backend.
   */
  Runtime();

  /**
   * Destructor. Releases all compute resources.
   */
  ~Runtime();

  // Non-copyable
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  // Movable
  Runtime(Runtime &&) noexcept;
  Runtime &operator=(Runtime &&) noexcept;

  /**
   * Checks if Vulkan backend is available on this system.
   * @return true if Vulkan is available, false otherwise.
   */
  bool isVulkanAvailable();

  /**
   * Initializes the compute backend.
   * @param backend The backend type to use (Vulkan or CPU).
   * @param numThreads Number of worker threads for CPU backend (0 = auto).
   * @param simdMode SIMD execution mode for CPU backend.
   */
  void init(BackendType backend = BackendType::CPU,
            size_t numThreads = 0,
            SIMDMode simdMode = SIMDMode::Auto);

  /**
   * Shuts down the runtime and releases all resources.
   */
  void shutdown();

  /**
   * Returns the current backend type.
   */
  BackendType currentBackend() const { return backendType_; }

  /**
   * Returns the number of worker threads (CPU backend only).
   */
  size_t numThreads() const;

  /**
   * Returns the current SIMD mode (CPU backend only).
   */
  SIMDMode simdMode() const;

  /**
   * Sets the SIMD mode for CPU backend.
   * @param mode The SIMD mode to use.
   */
  void setSIMDMode(SIMDMode mode);

  // =========================================================================
  // Buffer Operations
  // =========================================================================

  /**
   * Creates a buffer with the specified shape and data type.
   * @param shape Tensor shape (e.g., {batch, height, width, channels}).
   * @param dtype Data type of elements.
   * @param srcPtr Optional source data pointer for initialization.
   * @param isUniform If true, creates a uniform buffer (Vulkan only).
   * @return Handle to the created buffer.
   */
  ComputeHandle createBuffer(const std::vector<uint32_t> &shape,
                             DataType dtype,
                             const void *srcPtr = nullptr,
                             bool isUniform = false);

  /**
   * Creates an empty buffer with the specified shape and data type.
   * @param shape Tensor shape.
   * @param dtype Data type of elements.
   * @param isUniform If true, creates a uniform buffer (Vulkan only).
   * @return Handle to the created buffer.
   */
  ComputeHandle createBufferEmpty(const std::vector<uint32_t> &shape,
                                  DataType dtype,
                                  bool isUniform = false);

  /**
   * Copies data from host memory to a buffer.
   * @param handle Buffer handle.
   * @param srcPtr Source data pointer.
   * @param size Number of bytes to copy.
   * @param srcOffset Offset in source data.
   * @param dstOffset Offset in destination buffer.
   */
  void copyToBuffer(ComputeHandle handle,
                    const void *srcPtr,
                    size_t size,
                    size_t srcOffset = 0,
                    size_t dstOffset = 0);

  /**
   * Copies data from a buffer to host memory.
   * Automatically flushes any pending GPU commands before reading.
   * @param handle Buffer handle.
   * @param dstPtr Destination data pointer.
   * @param size Number of bytes to copy.
   * @param srcOffset Offset in source buffer.
   * @param dstOffset Offset in destination data.
   */
  void copyFromBuffer(ComputeHandle handle,
                      void *dstPtr,
                      size_t size,
                      size_t srcOffset = 0,
                      size_t dstOffset = 0);

  // =========================================================================
  // Operator Execution
  // =========================================================================

  /**
   * Executes a compute operator with the provided bindings.
   * This is the main entry point for executing compute operations.
   *
   * @param op The operator to execute (from OperatorEnum).
   * @param bindings Vector of compute bindings (buffers and data).
   * @param workgroupSize The workgroup/dispatch size for the operation.
   * @param dtype Data type for the operation.
   */
  void executeOperator(OperatorEnum op,
                       const std::vector<ComputeBinding> &bindings,
                       const ThreadSize &workgroupSize,
                       DataType dtype = DataType::Float32);

  /**
   * Encodes a compute operator using the Dispatcher.
   * Infers dtype and workgroup size from buffer bindings.
   *
   * @param op The operator to execute (from OperatorEnum).
   * @param bindings Vector of compute bindings (buffers and data).
   */
  void encodeOperator(OperatorEnum op,
                      const std::vector<ComputeBinding> &bindings);

  /**
   * Creates a shader/kernel for the specified operator.
   * @param op The operator type.
   * @param dtype Data type for the operation.
   * @return Handle to the created shader/kernel.
   */
  ComputeHandle createShader(OperatorEnum op,
                             DataType dtype = DataType::Float32);

  /**
   * Creates a shader module from SPIR-V bytecode (Vulkan only).
   * @param spirv SPIR-V bytecode.
   * @return Handle to the created shader module.
   */
  ComputeHandle createShaderModule(const std::vector<uint32_t> &spirv);

  // =========================================================================
  // Deferred Execution Support
  // =========================================================================

  /**
   * Encodes a dispatch and handles submission based on backend type.
   * For async backends (Vulkan): queues the dispatch and marks pending.
   * For sync backends (CPU): encodes, submits, and waits immediately.
   */
  void encodeAndMaybeSubmit(ComputeDispatch &&dispatch);

private:
  BackendType backendType_ = BackendType::CPU;
  std::shared_ptr<VulkanInstance> vulkanInstance_;
  std::unique_ptr<ComputeInterface> interface_;
  SIMDMode simdMode_ = SIMDMode::Auto;
  size_t numThreads_ = 0;
  bool vulkanAvailable_ = false;
  bool vulkanChecked_ = false;
  bool pendingCommands_ = false;

  // Shader cache: maps makeCacheKey(OperatorEnum, DataType) -> ComputeHandle
  std::map<uint64_t, ComputeHandle> shaderCache_;

  // Dispatcher for encoding operators
  std::unique_ptr<Dispatcher> dispatcher_;

  /**
   * Returns the underlying compute interface.
   * @throws std::runtime_error if not initialized.
   */
  ComputeInterface *getInterface();

  /**
   * Gets or creates a cached shader for the given operator and data type.
   */
  ComputeHandle getOrCreateShader(OperatorEnum op, DataType dtype);

  // =========================================================================
  // Deferred Execution Support
  // =========================================================================

  /**
   * Returns true if the current backend is a GPU backend.
   * GPU backends (Vulkan) execute commands lazily for better batching.
   * CPU backend executes commands immediately.
   */
  bool isGpuBackend() const { return backendType_ == BackendType::Vulkan; }

  /**
   * Flushes any pending commands by submitting and waiting.
   * No-op if there are no pending commands.
   */
  void flushPendingCommands();

  // =========================================================================
  // Operator Execution
  // =========================================================================

  /**
   * Submits the command buffer for execution.
   * @return Handle to the submitted command buffer.
   */
  ComputeHandle submit();

  /**
   * Waits for a command buffer to complete execution.
   * @param cmdBuffer Handle to the command buffer.
   */
  void wait(ComputeHandle cmdBuffer);
};

} // namespace cut
