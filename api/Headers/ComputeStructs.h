#pragma once

#include <ComputeCommon.h>

#include <string_view>

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

  /// Returns the binding index in the shader.
  uint32_t index() const { return bindingIndex; }

  /// Checks if this binding contains a ComputeHandle.
  bool isHandle() const { return type == Type::Handle; }

  /// Checks if this binding contains data.
  bool isData() const { return type == Type::Data; }

  /// Returns the bound handle (only valid if isHandle() is true).
  const ComputeHandle &getHandle() const { return handle; }

  /// Returns the bound data (only valid if isData() is true).
  const std::vector<uint8_t> &getData() const { return data; }

private:
  /// Indicates whether the binding contains a ComputeHandle or data.
  enum class Type { Handle, Data };

  ComputeHandle handle;      ///< Handle to a buffer or texture.
  std::vector<uint8_t> data; ///< Owned data (push constants).

  uint32_t bindingIndex; ///< The binding index in the shader.
  Type type;             ///< Indicates which member is active.
};

/**
 * Base struct for compute buffers across all backends.
 * Contains common metadata shared by all buffer implementations.
 */
struct ComputeBuffer {
  void *data = nullptr; ///< Pointer to mapped/accessible data.

  virtual ~ComputeBuffer() = default;

  /**
   * Returns the shape of the buffer.
   * Shape is always padded to exactly 4 dimensions.
   */
  const std::vector<uint32_t> getShape() const { return shape_; }

  /**
   * Sets the shape of the buffer.
   * If size < 4, pads with 1s at the end.
   * @param newShape The new shape (must have size <= 4).
   * @throws std::runtime_error if newShape.size() > 4.
   */
  void setShape(const std::vector<uint32_t> &newShape);

  /**
   * Returns the data type of the buffer elements.
   */
  DataType getDtype() const { return dtype_; }

  /**
   * Sets the data type of the buffer elements.
   * Recalculates buffer size if shape is already set.
   * @param newDtype The new data type.
   */
  void setDtype(DataType newDtype);

  /**
   * Returns dimension data for shader uniforms.
   * Format: [dim0, dim1, dim2, dim3] padded to 4 elements.
   */
  std::vector<uint32_t> getDimData() const;

  /**
   * Returns the total number of elements for execution.
   * The innermost dimension is aligned to a multiple of 4.
   */
  size_t executionSize() const;

  /**
   * Returns the buffer size in bytes (aligned size).
   */
  size_t size() const { return size_; }

  /**
   * Returns the size of the innermost dimension (before alignment).
   */
  uint32_t innerDimSize() const;

  /**
   * Calculates the actual buffer size in bytes (no padding).
   * Uses the buffer's shape and dtype.
   * @return Total size in bytes without alignment padding.
   */
  size_t calculateActualSize() const;

  /**
   * Calculates the total buffer size in bytes with alignment.
   * Rounds the innermost dimension up to a multiple of 4 for alignment.
   * Uses the buffer's shape and dtype.
   * @return Total size in bytes after aligning the innermost dimension.
   */
  size_t calculateAlignedSize() const {
    return executionElementCount_ * dataTypeSize(dtype_);
  }

  /**
   * Calculates the actual buffer size in bytes from a shape vector (no
   * padding).
   * @param shape Dimension-wise sizes (e.g., {batch, height, width, channels}).
   * @param dtype Data type of each element.
   * @return Total size in bytes without alignment padding.
   */
  static size_t calculateActualSize(const std::vector<uint32_t> &shape,
                                    DataType dtype);

  /**
   * Calculates the total buffer size in bytes from a shape vector.
   * Rounds the innermost dimension up to a multiple of 4 for alignment.
   * @param shape Dimension-wise sizes (e.g., {batch, height, width, channels}).
   * @param dtype Data type of each element.
   * @return Total size in bytes after aligning the innermost dimension.
   */
  static size_t calculateAlignedSize(const std::vector<uint32_t> &shape,
                                     DataType dtype);

private:
  std::vector<uint32_t> shape_; ///< Dimension-wise sizes (always size 4).
  DataType dtype_ = DataType::Float32; ///< Element data type.
  size_t size_ = 0;                    ///< Size in bytes.
  size_t executionElementCount_ = 0;
};

/**
 * Represents a single compute dispatch operation.
 * Contains shader, thread group size, workgroup size, and resource/data
 * bindings.
 */
class ComputeDispatch final {
public:
  /**
   * Constructs a ComputeDispatch with optional parameters.
   * @param shaderHandle Handle to the compute shader.
   * @param wgSize Workgroup dimensions for dispatch.
   * @param bindings Vector of compute bindings (handles or data references).
   */
  ComputeDispatch(const ComputeHandle &shaderHandle = {},
                  const ThreadSize &wgSize = {},
                  const std::vector<ComputeBinding> &bindings = {});

  /** Deleted copy constructor. */
  ComputeDispatch(const ComputeDispatch &) = delete;

  /** Default move constructor. */
  ComputeDispatch(ComputeDispatch &&) = default;

  /** Default move assignment operator. */
  ComputeDispatch &operator=(ComputeDispatch &&) = default;

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
   * Binds a scalar value to this dispatch.
   * The value is copied and owned by this dispatch.
   * @tparam T The type of the value (e.g., uint32_t, int32_t, float).
   * @param value The scalar value to bind.
   * @param bindIndex The binding index in the shader.
   */
  template <typename T>
  void bindValue(T value, uint32_t bindIndex) {
    DataReference ref(&value, sizeof(T));
    bindData(ref, bindIndex);
  }

  /**
   * Sets the workgroup size for this dispatch.
   * @param wgSize The workgroup dimensions.
   */
  void setWorkgroupSize(const ThreadSize &wgSize);

  /// Returns the workgroup size for this dispatch.
  const ThreadSize &workgroupSize() const;

  /// Returns the shader handle bound to this dispatch.
  const ComputeHandle &shader() const;

  /// Returns the list of resource bindings for this dispatch.
  const std::vector<ComputeBinding> &bindings() const;

  /// Sorts the bindings by their binding index.
  void sortBindings();

private:
  ThreadSize wgSize_;                    ///< Workgroup dimensions.
  ComputeHandle shader_;                 ///< Bound shader handle.
  std::vector<ComputeBinding> bindings_; ///< All bindings (handles and data).
};

/**
 * Represents a command buffer that records a sequence of compute dispatches.
 * Dispatches are executed in the order they are encoded.
 */
class CommandBuffer {
public:
  static constexpr std::string_view Name = "CommandBuffer";

  /**
   * Virtual destructor to ensure proper cleanup of derived classes.
   */
  virtual ~CommandBuffer() = default;

  /**
   * Begins recording commands to this command buffer.
   * Must be called before encoding any dispatches.
   * Default implementation does nothing; backends may override.
   */
  virtual void begin() {}

  /**
   * Encodes a compute dispatch to this command buffer.
   * @param dispatch The compute dispatch to encode.
   */
  void encode(ComputeDispatch &&dispatch);

  /**
   * Ends recording commands to this command buffer.
   * Must be called after all dispatches have been encoded and before submit.
   * Default implementation does nothing; backends may override.
   */
  virtual void end() {}

  /**
   * Submits the command buffer for execution on the GPU.
   * Ends recording and submits all encoded dispatches to the compute queue.
   */
  virtual void submit() = 0;

  /**
   * Waits for the command buffer to finish execution.
   * Blocks until all submitted commands have completed.
   */
  virtual void wait() = 0;

protected:
  /// Returns the list of encoded compute dispatches.
  const std::vector<ComputeDispatch> &dispatches() { return dispatches_; }

private:
  std::vector<ComputeDispatch> dispatches_; ///< List of compute dispatches.
};

} // namespace cut
