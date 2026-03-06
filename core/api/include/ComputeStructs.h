#pragma once

#include <ComputeCommon.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cut {

/** Forward declarations. */
class ComputeInterface;

/**
 * Holds either a ComputeHandle (for buffers/textures) or owned data,
 * without any binding index. Used as the payload inside ComputeBinding
 * and anywhere binding-index-free data is needed.
 */
class ComputeData final {
public:
  /**
   * Constructs a ComputeData with a ComputeHandle.
   * @param handleRef The ComputeHandle to hold.
   */
  ComputeData(const ComputeHandle &handleRef);

  /**
   * Constructs a ComputeData with a DataReference.
   * If the data fits in 4 bytes or fewer, it is stored inline as a scalar.
   * Otherwise, the data is copied into a vector.
   * @param dataRef The DataReference to copy data from.
   */
  ComputeData(const DataReference &dataRef);

  template <typename T>
  ComputeData(const T &dataRef) : ComputeData(DataReference(dataRef)) {}

  /// Checks if this holds a ComputeHandle.
  bool isHandle() const { return type == Type::Handle; }

  /// Checks if this holds vector data (size > 4 bytes).
  bool isData() const { return type == Type::Data; }

  /// Checks if this holds an inline scalar (size <= 4 bytes).
  bool isScalar() const { return type == Type::Scalar; }

  /// Returns the held handle (only valid if isHandle() is true).
  const ComputeHandle &getHandle() const;

  /// Returns the held data (only valid if isData() is true).
  const std::vector<uint8_t> &getData() const;

  /// Returns the held scalar cast to T (only valid if isScalar() is true).
  template <typename T>
  T getScalar() const {
    static_assert(sizeof(T) <= sizeof(Scalar),
                  "T must be <= sizeof(Scalar) (4 bytes)");
    if (type != Type::Scalar) {
      logErr("getScalar() called on non-Scalar ComputeData");
    }
    T result;
    std::memcpy(&result, &scalar, sizeof(T));
    return result;
  }

private:
  /// Scalar union that can hold either a uint32_t or a float.
  union Scalar {
    uint32_t u;
    float f;
  };

  /// Indicates whether this holds a ComputeHandle, data, or a scalar.
  enum class Type { Handle, Data, Scalar };

  ComputeHandle handle;      ///< Handle to a buffer or texture.
  std::vector<uint8_t> data; ///< Owned data (push constants, > 4 bytes).
  Scalar scalar = {};        ///< Inline scalar (<= 4 bytes).
  Type type;                 ///< Indicates which member is active.
};

/**
 * Represents a binding for a compute dispatch operation.
 * Pairs a binding index with a ComputeData payload.
 */
class ComputeBinding final {
public:
  /**
   * Constructs a ComputeBinding with a ComputeHandle.
   * @param index The binding index in the shader.
   * @param handleRef The ComputeHandle to bind.
   */
  ComputeBinding(uint32_t index, const ComputeHandle &handleRef)
      : bindingIndex(index), payload(handleRef) {}

  /**
   * Constructs a ComputeBinding with a DataReference.
   * The data is copied and owned by this binding.
   * @param index The binding index in the shader.
   * @param dataRef The DataReference to copy data from.
   */
  ComputeBinding(uint32_t index, const DataReference &dataRef)
      : bindingIndex(index), payload(dataRef) {}

  /// Returns the binding index in the shader.
  uint32_t index() const { return bindingIndex; }

  /// Checks if this binding contains a ComputeHandle.
  bool isHandle() const { return payload.isHandle(); }

  /// Checks if this binding contains data.
  bool isData() const { return payload.isData(); }

  /// Checks if this binding contains an inline scalar.
  bool isScalar() const { return payload.isScalar(); }

  /// Returns the bound handle (only valid if isHandle() is true).
  const ComputeHandle &getHandle() const { return payload.getHandle(); }

  /// Returns the bound data (only valid if isData() is true).
  const std::vector<uint8_t> &getData() const { return payload.getData(); }

  /// Returns the bound scalar cast to T (only valid if isScalar() is true).
  template <typename T>
  T getScalar() const {
    return payload.getScalar<T>();
  }

  /// Returns the underlying ComputeData payload.
  const ComputeData &computeData() const { return payload; }

private:
  ComputeData payload;   ///< The handle or data payload.
  uint32_t bindingIndex; ///< The binding index in the shader.
};

/**
 * Base struct for compute buffers across all backends.
 * Contains common metadata shared by all buffer implementations.
 *
 * Shape convention:
 *   shape_.size() defines the total number of dimensions.
 *   shape_[0] is the most significant (outermost) dimension.
 *   shape_[shape_.size() - 1] is the least significant (innermost) dimension.
 *   e.g. for a 3D buffer: shape_[0] = Z, shape_[1] = Y, shape_[2] = X.
 *
 * Alignment:
 *   The least significant (innermost) dimension is always aligned to a
 *   multiple of 4 when allocating memory and storing data. This must be
 *   considered in shaders, dispatch sizing, copy operations, and any code
 *   that computes offsets or strides.
 */
struct ComputeBuffer {
  void *data = nullptr; ///< Pointer to mapped/accessible data.

  virtual ~ComputeBuffer() = default;

  /**
   * Returns the shape of the buffer.
   */
  const std::vector<uint32_t> getShape() const { return shape_; }

  /**
   * Sets the shape of the buffer.
   * @param newShape The new shape (must have 1-4 dimensions).
   * @throws std::runtime_error if newShape is empty or size > 4.
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

  /**
   * Calculates the total number of elements with the innermost dimension
   * aligned to a multiple of 4.
   * @param shape Dimension-wise sizes (e.g., {batch, height, width, channels}).
   * @return Total element count with aligned innermost dimension.
   */
  static size_t calculateAlignedElements(const std::vector<uint32_t> &shape);

  /**
   * Infers the data type from compute bindings.
   * Looks for the first buffer binding and returns its dtype.
   * Validates that all buffer bindings have the same dtype.
   * @tparam F Callable type that takes ComputeHandle and returns const
   * ComputeBuffer&.
   * @param bindings Vector of compute bindings.
   * @param getBuffer Function to get a ComputeBuffer reference from a handle.
   * @return The inferred data type, or Float32 if no buffer bindings found.
   * @throws std::runtime_error if buffer dtypes don't match.
   */
  template <typename F>
  static DataType inferDataType(const std::vector<ComputeBinding> &bindings,
                                F &&getBuffer);

private:
  std::vector<uint32_t> shape_; ///< Dimension-wise sizes (1-4 dimensions).
  DataType dtype_ = DataType::Float32; ///< Element data type.
  size_t size_ = 0;                    ///< Size in bytes.
  size_t executionElementCount_ = 0;
};

template <typename F>
DataType
ComputeBuffer::inferDataType(const std::vector<ComputeBinding> &bindings,
                             F &&getBuffer) {
  DataType inferredDtype = DataType::Float32;
  bool dtypeSet = false;

  for (const auto &binding : bindings) {
    if (!binding.isHandle()) {
      continue;
    }

    const ComputeBuffer &buffer = getBuffer(binding.getHandle());

    if (!dtypeSet) {
      inferredDtype = buffer.getDtype();
      dtypeSet = true;
    } else if (buffer.getDtype() != inferredDtype) {
      throw std::runtime_error(std::string("Buffer dtype mismatch: expected ") +
                               dataTypeName(inferredDtype) + " but got " +
                               dataTypeName(buffer.getDtype()));
    }
  }

  return inferredDtype;
}

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

  /**
   * Creates a barrier-only dispatch that inserts a compute-to-compute memory
   * barrier. No shader is dispatched; instead, a pipeline barrier is recorded
   * to ensure prior shader writes are visible to subsequent shader reads.
   */
  static ComputeDispatch createBarrier();

  /**
   * Returns true if this dispatch is a barrier (no shader dispatch).
   */
  bool isBarrier() const { return isBarrier_; }

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

  /// Sets a human-readable label for profiling.
  void setLabel(std::string label) { label_ = std::move(label); }

  /// Returns the label for this dispatch.
  const std::string &label() const { return label_; }

private:
  ThreadSize wgSize_;                    ///< Workgroup dimensions.
  ComputeHandle shader_;                 ///< Bound shader handle.
  std::vector<ComputeBinding> bindings_; ///< All bindings (handles and data).
  bool isBarrier_ = false; ///< True if this is a barrier-only dispatch.
  std::string label_;      ///< Human-readable label for profiling.
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

  /**
   * Re-submits a previously recorded command buffer without re-recording.
   * Waits for any prior execution to complete, then submits again.
   * Only valid for reusable command buffers.
   */
  virtual void resubmit() {}

  /// Marks this command buffer as reusable (can be re-submitted).
  void setReusable(bool reusable) { reusable_ = reusable; }

  /// Returns true if this command buffer is reusable.
  bool isReusable() const { return reusable_; }

  /// Enables or disables per-dispatch GPU profiling for this command buffer.
  void setProfilingEnabled(bool enabled) { profilingEnabled_ = enabled; }

  /// Returns true if GPU profiling is enabled.
  bool isProfilingEnabled() const { return profilingEnabled_; }

protected:
  /// Returns the list of encoded compute dispatches.
  const std::vector<ComputeDispatch> &dispatches() { return dispatches_; }

  bool profilingEnabled_ = false; ///< Per-dispatch GPU profiling flag.
  bool reusable_ = false;         ///< Whether this CB can be re-submitted.

private:
  std::vector<ComputeDispatch> dispatches_; ///< List of compute dispatches.
};

using Tensor = ComputeHandle;
using TensorLike = ComputeData;

} // namespace cut
