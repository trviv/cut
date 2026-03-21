#pragma once

#include "ComputeCommon.h"
#include "ComputeStructs.h"
#include <ComputeContainers.h>
#include <ComputeHandle.h>

#include <memory>

namespace cut {

/**
 * Abstract base class for compute API interfaces.
 * Provides common functionality for buffer management, shader creation,
 * and dispatch operations across different compute backends.
 */
class ComputeInterface {
public:
  /**
   * Default constructor.
   * Derived classes must call setCommandBufferContainer() to initialize
   * the command buffer container before using command buffer operations.
   */
  ComputeInterface() = default;

  /** Virtual destructor. */
  virtual ~ComputeInterface() = default;

  /** Deleted copy constructor. */
  ComputeInterface(const ComputeInterface &) = delete;

  /** Deleted move constructor. */
  ComputeInterface(ComputeInterface &&) = delete;

  /**
   * Creates a GPU buffer with tensor-like shape.
   * The innermost dimension is rounded up to a multiple of 4 for alignment.
   * @param shape Dimension-wise sizes (e.g., {batch, height, width, channels}).
   * @param dtype Data type of each element.
   * @param hostSourcePtr Optional pointer to host data for initialization.
   * @param immutable If true, buffer contents cannot be modified after
   * creation.
   * @return Handle to the created buffer.
   */
  virtual ComputeHandle createBuffer(const std::vector<uint32_t> &shape,
                                     DataType dtype,
                                     const void *hostSourcePtr = nullptr,
                                     bool immutable = false) = 0;

  /**
   * Creates a host-visible, coherent buffer for CPU-GPU shared data.
   * Writes via copyDataToBuffer go directly to mapped memory (no staging,
   * no fence wait). Ideal for small, frequently-updated per-token params.
   */
  virtual ComputeHandle
  createBufferMapped(const std::vector<uint32_t> &shape,
                     DataType dtype,
                     const void *hostSourcePtr = nullptr) {
    // Default implementation falls back to regular buffer.
    return createBuffer(shape, dtype, hostSourcePtr, false);
  }

  /**
   * Creates a buffer view referencing a sub-region of an existing buffer.
   * The view shares the parent's GPU buffer but binds at a non-zero offset.
   * The parent is kept alive via ref-counting.
   * @param parent Handle to the parent buffer.
   * @param byteOffset Byte offset into the parent buffer (must be aligned to
   *                   device storage buffer offset alignment).
   * @param shape Shape of the view.
   * @param dtype Data type of view elements.
   * @return Handle to the created buffer view.
   */
  virtual ComputeHandle createBufferView(const ComputeHandle &parent,
                                         size_t byteOffset,
                                         const std::vector<uint32_t> &shape,
                                         DataType dtype) = 0;

  /**
   * Copies data from host memory to a GPU buffer.
   * @param hostSourcePtr Pointer to the source data in host memory.
   * @param destBufferHandle Handle to the destination GPU buffer.
   * @param copySize Number of bytes to copy.
   * @param srcOffset Offset in the source data.
   * @param destOffset Offset in the destination buffer.
   * @param useStagingBuffer If true, use a staging buffer for the transfer.
   * @param waitForCompletion If true, block until copy completes.
   */
  virtual void copyDataToBuffer(const void *hostSourcePtr,
                                const ComputeHandle &destBufferHandle,
                                size_t copySize,
                                size_t srcOffset,
                                size_t destOffset,
                                bool useStagingBuffer = false,
                                bool waitForCompletion = false) = 0;

  /**
   * Copies data from a GPU buffer to host memory.
   * @param srcBufferHandle Handle to the source GPU buffer.
   * @param hostDestPtr Pointer to the destination in host memory.
   * @param copySize Number of bytes to copy.
   * @param srcOffset Offset in the source buffer.
   * @param destOffset Offset in the destination data.
   * @param useStagingBuffer If true, use a staging buffer for the transfer.
   * @param waitForCompletion If true, block until copy completes.
   */
  virtual void copyDataFromBuffer(const ComputeHandle &srcBufferHandle,
                                  void *hostDestPtr,
                                  size_t copySize,
                                  size_t srcOffset,
                                  size_t destOffset,
                                  bool useStagingBuffer = false,
                                  bool waitForCompletion = false) = 0;

  /**
   * Creates a shader module from SPIR-V bytecode.
   * @param spirvCode Vector containing the SPIR-V bytecode.
   * @return Handle to the created shader module.
   */
  virtual ComputeHandle
  createShaderModule(const std::vector<uint32_t> &spirvCode) = 0;

  /**
   * Returns a const reference to the buffer metadata.
   * @param bufferHandle Handle to the buffer.
   * @return Const reference to the ComputeBuffer containing shape, dtype, etc.
   */
  virtual const ComputeBuffer &
  getBuffer(const ComputeHandle &bufferHandle) const = 0;

  /**
   * Returns the number of active (in-use) buffers.
   */
  virtual size_t bufferCount() const = 0;

  /**
   * Returns total GPU memory actively allocated for buffers (excludes views).
   */
  virtual size_t activeBufferMemoryBytes() const { return 0; }

  /**
   * Returns the minimum byte alignment required for buffer view offsets.
   * Defaults to 256 (safe upper bound for most devices).
   */
  virtual size_t bufferOffsetAlignment() const { return 256; }

  /**
   * Release internal caches and staging memory to reduce memory footprint.
   * Call after bulk loading (e.g. model weights) is complete.
   */
  virtual void releaseLoadingResources() {}

  /**
   * Flushes any pending batched host-to-device transfers.
   * Called automatically before compute submission and device-to-host reads.
   * Override in backends that batch staging buffer uploads.
   */
  virtual void flushTransfers() {}

  /**
   * Enables or disables per-dispatch GPU profiling.
   * When enabled, command buffers will record hardware timestamps around each
   * dispatch and log per-operation timing after execution completes.
   */
  void setProfilingEnabled(bool enabled);

  /**
   * Encodes a compute dispatch to the active command buffer.
   * If no command buffer is currently recording, one will be created.
   * @param dispatch The compute dispatch object to encode (moved).
   */
  void encode(ComputeDispatch &&dispatch);

  /**
   * Ends recording and submits the command buffer for GPU execution.
   * @return Handle to the submitted command buffer.
   */
  ComputeHandle submit();

  /**
   * Ends recording and submits the command buffer marked as reusable.
   * The returned handle can be passed to resubmit() for subsequent executions.
   * @return Handle to the submitted reusable command buffer.
   */
  ComputeHandle submitReusable();

  /**
   * Re-submits a previously recorded reusable command buffer.
   * Waits for prior execution, resets fence, and re-submits without
   * re-recording.
   * @param commandBufferHandle Handle to the reusable command buffer.
   */
  void resubmit(const ComputeHandle &commandBufferHandle);

  /**
   * Waits for a command buffer to finish execution.
   * Blocks until all submitted commands have completed.
   * @param commandBufferHandle Handle to the command buffer to wait on.
   */
  void wait(const ComputeHandle &commandBufferHandle);

protected:
  /**
   * Copies data from actual-sized host memory to aligned buffer memory.
   * Handles row-by-row copying when the innermost dimension needs padding.
   * For partial copies (non-zero offsets or size != actualSize), uses memcpy.
   * @param src Pointer to the source data (actual size).
   * @param dst Pointer to the destination buffer (aligned size).
   * @param buffer The ComputeBuffer containing shape and dtype info.
   * @param srcOffset Byte offset in the source data.
   * @param dstOffset Byte offset in the destination buffer.
   * @param size Number of bytes to copy (0 = full buffer).
   */
  static void copyActualToAligned(const void *src,
                                  void *dst,
                                  const ComputeBuffer &buffer,
                                  size_t srcOffset = 0,
                                  size_t dstOffset = 0,
                                  size_t size = 0);

  /**
   * Copies data from aligned buffer memory to actual-sized host memory.
   * Handles row-by-row copying when the innermost dimension has padding.
   * For partial copies (non-zero offsets or size != actualSize), uses memcpy.
   * @param src Pointer to the source buffer (aligned size).
   * @param dst Pointer to the destination data (actual size).
   * @param buffer The ComputeBuffer containing shape and dtype info.
   * @param srcOffset Byte offset in the source buffer.
   * @param dstOffset Byte offset in the destination data.
   * @param size Number of bytes to copy (0 = full buffer).
   */
  static void copyAlignedToActual(const void *src,
                                  void *dst,
                                  const ComputeBuffer &buffer,
                                  size_t srcOffset = 0,
                                  size_t dstOffset = 0,
                                  size_t size = 0);

  /**
   * Sets the command buffer container for this interface.
   * Must be called by derived classes during construction.
   * @param commandBufferContainer Unique pointer to the container.
   */
  void setCommandBufferContainer(
      std::unique_ptr<CommandBufferContainer> commandBufferContainer);

private:
  ///< Container for command buffers.
  std::unique_ptr<CommandBufferContainer> commandBufferContainer_;

  ///< Currently recording command buffer handle.
  ComputeHandle activeCommandBuffer_;

  bool profilingEnabled_ = false; ///< Per-dispatch GPU profiling flag.
};

} // namespace cut
