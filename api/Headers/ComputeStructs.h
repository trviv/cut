#pragma once

#include <ComputeCommon.h>

namespace cut {

/** Forward declarations. */
class ComputeInterface;

/**
 * Represents a binding for a compute dispatch operation.
 * Can hold either a ComputeHandle (for buffers/textures) or owned data.
 * Data is copied and owned by this struct.
 */
class ComputeBinding final {
public:
  /**
   * Constructs a ComputeBinding with a ComputeHandle.
   * @param index The binding index in the shader.
   * @param handleRef The ComputeHandle to bind.
   */
  ComputeBinding(uint32_t index, const ComputeHandle &handleRef)
      : bindingIndex(index), type(Type::Handle), handle(handleRef) {}

  /**
   * Constructs a ComputeBinding with a DataReference.
   * The data is copied and owned by this binding.
   * @param index The binding index in the shader.
   * @param dataRef The DataReference to copy data from.
   */
  ComputeBinding(uint32_t index, const DataReference &dataRef)
      : bindingIndex(index), type(Type::Data) {
    const auto *bytePtr = static_cast<const uint8_t *>(dataRef.ptr);
    data = {bytePtr, bytePtr + dataRef.size};
  }

private:
  /**
   * Indicates whether the binding contains a ComputeHandle or data.
   */
  enum class Type { Handle, Data };

  /**
   * Checks if this binding contains a ComputeHandle.
   * @return True if the binding holds a ComputeHandle.
   */
  bool isHandle() const { return type == Type::Handle; }

  /**
   * Checks if this binding contains data.
   * @return True if the binding holds data.
   */
  bool isData() const { return type == Type::Data; }

  ComputeHandle handle;      ///< Handle to a buffer or texture.
  std::vector<uint8_t> data; ///< Owned data (push constants).

  uint32_t bindingIndex; ///< The binding index in the shader.
  Type type;             ///< Indicates which member is active.
};

/**
 * Represents a single compute dispatch operation.
 * Contains shader, thread group size, and resource/data bindings.
 */
class ComputeDispatch final {
public:
  /**
   * Constructs a ComputeDispatch with optional parameters.
   * @param shaderHandle Handle to the compute shader.
   * @param tgSize Thread group dimensions for dispatch.
   * @param bindings Vector of compute bindings (handles or data references).
   */
  ComputeDispatch(const ComputeHandle &shaderHandle = {},
                  const ThreadGroupSize &tgSize = {},
                  const std::vector<ComputeBinding> &bindings = {});

  /** Deleted copy constructor. */
  ComputeDispatch(const ComputeDispatch &) = delete;

  /** Default move constructor. */
  ComputeDispatch(ComputeDispatch &&) = default;

  /** Default move assignment operator. */
  ComputeDispatch &operator=(ComputeDispatch &&) = default;

private:
  friend class ComputeInterface;

  /**
   * Binds a shader to this dispatch.
   * @param shaderHandle Handle to the compute shader.
   */
  void bindShader(const ComputeHandle &shaderHandle);

  /**
   * Binds a resource (buffer or texture) to this dispatch.
   * @param resourceHandle Handle to the buffer or texture.
   * @param bindIndex The binding index in the shader.
   */
  void bindResource(const ComputeHandle &resourceHandle, uint32_t bindIndex);

  /**
   * Binds raw data to this dispatch.
   * @param dataRef Reference to the data to bind.
   * @param bindIndex The binding index in the shader.
   */
  void bindData(const DataReference &dataRef, uint32_t bindIndex);

  /**
   * Sets the thread group size for this dispatch.
   * @param tgSize The thread group dimensions.
   */
  void setThreadGroupSize(const ThreadGroupSize &tgSize);

  ThreadGroupSize tgSize_;               ///< Thread group dimensions.
  ComputeHandle shader_;                 ///< Bound shader handle.
  std::vector<ComputeBinding> bindings_; ///< All bindings (handles and data).
};

/**
 * Represents a command buffer that records a sequence of compute dispatches.
 * Dispatches are executed in the order they are encoded.
 */
class CommandBuffer {
public:
  /**
   * Virtual destructor to ensure proper cleanup of derived classes.
   */
  virtual ~CommandBuffer() = default;

  /**
   * Encodes a dispatch handle to this command buffer.
   * @param dispatchHandle Handle to the compute dispatch to encode.
   */
  void encode(const ComputeHandle &dispatchHandle);

protected:
  /**
   * Backend-specific implementation for encoding a dispatch.
   * This pure virtual function must be implemented by derived classes to
   * perform backend-specific work when a dispatch is encoded. Implementations
   * can use this to record GPU commands, create pipeline state objects,
   * bind resources, validate dispatch parameters, or perform any other
   * backend-specific preparation for execution.
   *
   * Called by encode() after the dispatch handle has been added to the
   * internal dispatch list. The dispatch object is passed by const reference
   * to allow implementations to inspect shader bindings, thread group sizes,
   * and other dispatch configuration without modifying the dispatch.
   *
   * @param dispatch Const reference to the compute dispatch being encoded.
   */
  virtual void encodeImpl(const ComputeDispatch &dispatch) = 0;

private:
  std::vector<ComputeHandle> dispatches_; ///< List of dispatch handles.
};

} // namespace cut
