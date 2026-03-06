#pragma once

#include "TensorStore.h"

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeInterface.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

#include <memory>
#include <utility>
#include <vector>

namespace cut {

// Forward declarations
class VulkanInstance;
class VulkanCompute;
class Dispatcher;
class Operations;
class OpNode;

/**
 * Backend type enum for runtime selection.
 */
enum class BackendType { Vulkan };

/**
 * Runtime class that manages compute backend lifecycle and operator execution.
 * Provides a unified interface for executing compute operations on Vulkan GPU.
 * All compute operations should be issued through the Operations object
 * returned by ops().
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

  // Non-movable
  Runtime(Runtime &&) = delete;
  Runtime &operator=(Runtime &&) = delete;

  /**
   * Checks if Vulkan backend is available on this system.
   * @return true if Vulkan is available, false otherwise.
   */
  bool isVulkanAvailable();

  /**
   * Initializes the compute backend.
   * @param backend The backend type to use (Vulkan).
   */
  void init(BackendType backend = BackendType::Vulkan);

  /**
   * Shuts down the runtime and releases all resources.
   */
  void shutdown();

  /**
   * Returns the current backend type.
   */
  BackendType currentBackend() const { return backendType_; }

  // =========================================================================
  // Tensor Operations
  // =========================================================================

  /**
   * Creates a tensor with the specified shape and data type.
   * @param shape Tensor shape (e.g., {batch, height, width, channels}).
   * @param dtype Data type of elements.
   * @param srcPtr Optional source data pointer for initialization.
   * @param isUniform If true, creates a uniform buffer (Vulkan only).
   * @return Handle to the created tensor.
   */
  Tensor createTensor(const std::vector<uint32_t> &shape,
                      DataType dtype,
                      const void *srcPtr = nullptr,
                      bool isUniform = false);

  /**
   * Creates an empty tensor with the specified shape and data type.
   * @param shape Tensor shape.
   * @param dtype Data type of elements.
   * @param isUniform If true, creates a uniform buffer (Vulkan only).
   * @return Handle to the created tensor.
   */
  Tensor createTensorEmpty(const std::vector<uint32_t> &shape,
                           DataType dtype,
                           bool isUniform = false);

  /**
   * Copies data from host memory to a tensor.
   * @param handle Tensor handle.
   * @param srcPtr Source data pointer.
   * @param size Number of bytes to copy.
   * @param srcOffset Offset in source data.
   * @param dstOffset Offset in destination tensor.
   */
  void copyToTensor(Tensor handle,
                    const void *srcPtr,
                    size_t size,
                    size_t srcOffset = 0,
                    size_t dstOffset = 0);

  /**
   * Returns buffer metadata (shape, dtype, size) for a tensor handle.
   * @param handle Tensor handle.
   * @return Const reference to the ComputeBuffer.
   */
  const ComputeBuffer &getTensor(const Tensor &handle) const;

  /**
   * Copies data from a tensor to host memory.
   * Automatically flushes any pending GPU commands before reading.
   * @param handle Tensor handle.
   * @param dstPtr Destination data pointer.
   * @param size Number of bytes to copy.
   * @param srcOffset Offset in source tensor.
   * @param dstOffset Offset in destination data.
   */
  void copyFromTensor(Tensor handle,
                      void *dstPtr,
                      size_t size,
                      size_t srcOffset = 0,
                      size_t dstOffset = 0);

  // =========================================================================
  // Operations
  // =========================================================================

  /**
   * Returns the number of active (in-use) GPU buffers.
   */
  size_t bufferCount() const;

  /**
   * Returns total GPU memory actively allocated for buffers (excludes views).
   */
  size_t activeBufferMemoryBytes() const;

  /**
   * Flushes any pending GPU commands.
   * Submits and waits for all batched operations to complete.
   */
  void flush();

  /**
   * Returns a reference to the Operations object for issuing compute
   * operations.
   * @throws std::runtime_error if not initialized.
   */
  Operations &ops();

  /**
   * Returns the TensorStore for buffer creation and metadata queries.
   */
  TensorStore &store();

  /**
   * Enables or disables per-dispatch GPU profiling.
   * When enabled, hardware timestamps are recorded around each dispatch
   * and per-operation timing is logged after execution completes.
   */
  void setProfilingEnabled(bool enabled);

  /**
   * Dispatches a compute operator using an OpNode.
   * The OpNode provides all operator-level information.
   */
  void dispatch(std::unique_ptr<OpNode> node);
  void dispatch(OpNode &node);

private:
  BackendType backendType_ = BackendType::Vulkan;
  std::shared_ptr<VulkanInstance> vulkanInstance_;
  std::unique_ptr<ComputeInterface> interface_;
  bool vulkanAvailable_ = false;
  bool vulkanChecked_ = false;
  bool pendingCommands_ = false;
  bool profilingEnabled_ = false;

  // Tensor buffer storage
  std::unique_ptr<TensorStore> store_;

  // Dispatcher for encoding operators
  std::unique_ptr<Dispatcher> dispatcher_;

  // High-level operations
  std::unique_ptr<Operations> operations_;

  /**
   * Returns the underlying compute interface.
   * @throws std::runtime_error if not initialized.
   */
  ComputeInterface *getInterface();

  bool isGpuBackend() const { return backendType_ == BackendType::Vulkan; }

  void flushPendingCommands();
};

} // namespace cut
