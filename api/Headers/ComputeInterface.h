#pragma once

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
  /** Default constructor. */
  explicit ComputeInterface() = default;

  /** Virtual destructor. */
  virtual ~ComputeInterface() = default;

  /** Deleted copy constructor. */
  ComputeInterface(const ComputeInterface &) = delete;

  /** Deleted move constructor. */
  ComputeInterface(ComputeInterface &&) = delete;

  /**
   * Creates a GPU buffer.
   * @param size Size of the buffer in bytes.
   * @param hostSourcePtr Optional pointer to host data for initialization.
   * @param immutable If true, buffer contents cannot be modified after
   * creation.
   * @return Handle to the created buffer.
   */
  virtual ComputeHandle createBuffer(size_t size,
                                     const void *hostSourcePtr = nullptr,
                                     bool immutable = false) = 0;

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
   * Begins recording commands to a new command buffer.
   * Must be paired with endCommandBuffer().
   */
  void beginCommandBuffer();

  /**
   * Ends recording and returns the completed command buffer.
   * @return Handle to the recorded command buffer.
   */
  ComputeHandle endCommandBuffer();

  /**
   * Encodes a compute dispatch to the active command buffer.
   * Must be called between beginCommandBuffer() and endCommandBuffer().
   * @param dispatch The compute dispatch object to encode (moved).
   * @return Handle to the encoded dispatch.
   */
  ComputeHandle encode(ComputeDispatch &&dispatch);

  /**
   * Submits a command buffer for GPU execution.
   * @param commandBufferHandle Handle to the command buffer to submit.
   */
  virtual void submit(const ComputeHandle &commandBufferHandle) = 0;

  /**
   * Returns the ComputeDispatchContainer associated with a handle (const).
   * @param handle The handle to the command buffer.
   * @return Const reference to the ComputeDispatchContainer.
   */
  const CommandBuffer &getCommandBuffer(const ComputeHandle &handle) const;

protected:
  /**
   * Returns the ComputeDispatchContainer associated with a handle.
   * @param handle The handle to the command buffer.
   * @return Reference to the ComputeDispatchContainer.
   */
  CommandBuffer &getCommandBuffer(const ComputeHandle &handle);

  /**
   * Creates a backend-specific CommandBuffer instance.
   * This pure virtual function must be implemented by derived classes to
   * instantiate their specific CommandBuffer implementations.
   * @return Unique pointer to the created CommandBuffer.
   */
  virtual std::unique_ptr<CommandBuffer> createCommandBuffer() = 0;

private:
  ///< Container for command buffers.
  CommandBufferContainer commandBufferContainer_;

  ///< Currently recording command buffer.
  std::unique_ptr<CommandBuffer> activeCommandBuffer_;
};

} // namespace cut
