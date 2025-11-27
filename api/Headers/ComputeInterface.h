#pragma once

#include <ComputeContainers.h>
#include <ComputeHandle.h>

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
                                size_t copySize, size_t srcOffset,
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
                                  void *hostDestPtr, size_t copySize,
                                  size_t srcOffset, size_t destOffset,
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
   * Creates a new compute dispatch with the specified parameters.
   * @param shaderHandle Handle to the compute shader.
   * @param tgSize Thread group dimensions for the dispatch.
   * @param bindings Vector of compute bindings (handles or data references).
   * @return Handle to the created dispatch.
   */
  ComputeHandle
  createDispatch(const ComputeHandle &shaderHandle = {},
                 const ThreadGroupSize &tgSize = {},
                 const std::vector<ComputeBinding> &bindings = {});

  /**
   * Creates a new dispatch that references an existing dispatch's settings.
   * @param refDispatchHandle Handle to the reference dispatch.
   * @return Handle to the created dispatch.
   */
  ComputeHandle createDispatchFromRef(const ComputeHandle &refDispatchHandle);

  /**
   * Binds a shader to an existing dispatch.
   * @param dispatchHandle Handle to the dispatch to modify.
   * @param shaderHandle Handle to the shader to bind.
   */
  void bindShader(const ComputeHandle &dispatchHandle,
                  const ComputeHandle &shaderHandle);

  /**
   * Binds a resource (buffer or texture) to an existing dispatch.
   * @param dispatchHandle Handle to the dispatch to modify.
   * @param resourceHandle Handle to the buffer or texture to bind.
   * @param bindIndex The binding index in the shader.
   */
  void bindResource(const ComputeHandle &dispatchHandle,
                    const ComputeHandle &resourceHandle, uint32_t bindIndex);

  /**
   * Binds multiple resources (buffers or textures) to an existing dispatch.
   * @param dispatchHandle Handle to the dispatch to modify.
   * @param resourceHandles Vector of handles to the buffers or textures to
   * bind.
   * @param bindOffset Starting binding index; resources are bound sequentially
   * from this offset.
   */
  void bindResources(const ComputeHandle &dispatchHandle,
                     const std::vector<ComputeHandle> &resourceHandles,
                     uint32_t bindOffset = 0);

  /**
   * Binds raw data (push constants) to an existing dispatch.
   * @param dispatchHandle Handle to the dispatch to modify.
   * @param dataRef Reference to the data to bind.
   * @param bindIndex The binding index in the shader.
   */
  void bindData(const ComputeHandle &dispatchHandle,
                const DataReference &dataRef, uint32_t bindIndex);

  /**
   * Sets the thread group size for an existing dispatch.
   * @param dispatchHandle Handle to the dispatch to modify.
   * @param tgSize The thread group dimensions.
   */
  void setThreadGroupSize(const ComputeHandle &dispatchHandle,
                          const ThreadGroupSize &tgSize);

  /**
   * Creates a dispatch list from a vector of dispatch handles.
   * @param dispatches Vector of dispatch handles to include in the list.
   * @return Handle to the created dispatch list.
   */
  ComputeHandle createDispatchList(DispatchList &&dispatches);

  /**
   * Submits a dispatch or dispatch list for GPU execution.
   * @param handle Handle to a dispatch or dispatch list.
   */
  virtual void submit(const ComputeHandle &handle) = 0;

protected:
  DispatchContainer dispatchContainer_; ///< Container for dispatch handles.
  DispatchListContainer
      dispatchListContainer_; ///< Container for dispatch list handles.
};

} // namespace cut
