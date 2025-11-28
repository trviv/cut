#pragma once

#include <ComputeCommon.h>
#include <ComputeContainer.h>

namespace cut {

/** Forward declarations. */
class ComputeInterface;

/**
 * A list of dispatch handles to be executed sequentially.
 */
using DispatchList = std::vector<ComputeHandle>;

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
 * @def CONTAINER_METHODS_IMPL(STRUCT_NAME)
 * Macro that generates standard container methods for a given struct type.
 * Generates: createHandle, get (mutable/const from handle), get (mutable/const
 * from data).
 */
#define CONTAINER_METHODS_IMPL(STRUCT_NAME)                                    \
  ComputeHandle createHandle(STRUCT_NAME &&structData) {                       \
    return ComputeDataContainer::createHandle(                                 \
        new STRUCT_NAME(std::move(structData)));                               \
  }                                                                            \
                                                                               \
  STRUCT_NAME *get(const ComputeHandle &handle) {                              \
    return data(handle).get<STRUCT_NAME *>();                                  \
  }                                                                            \
                                                                               \
  const STRUCT_NAME *get(const ComputeHandle &handle) const {                  \
    return data(handle).get<const STRUCT_NAME *>();                            \
  }                                                                            \
                                                                               \
  STRUCT_NAME *get(const HandleData &data) {                                   \
    return data.get<STRUCT_NAME *>();                                          \
  }                                                                            \
                                                                               \
  const STRUCT_NAME *get(const HandleData &data) const {                       \
    return data.get<const STRUCT_NAME *>();                                    \
  }

/**
 * @def CONTAINER_DELETE_METHOD_IMPL(STRUCT_NAME)
 * Macro that generates a deletion method for handle ID.
 * Safely deletes the pointer stored at the given ID if non-null.
 */
#define CONTAINER_DELETE_METHOD_IMPL(STRUCT_NAME)                              \
  void deleteHandlePtr(size_t id) {                                            \
    auto *ptr = objects_[id].get<STRUCT_NAME *>();                             \
    if (ptr != nullptr) {                                                      \
      delete ptr;                                                              \
    }                                                                          \
  }

/**
 * Represents a single compute dispatch operation.
 * Contains shader, thread group size, and resource/data bindings.
 */
class ComputeDispatch final {
private:
  friend class DispatchContainer;
  friend class ComputeInterface;

  /**
   * Constructs a ComputeDispatch with optional parameters.
   * @param shaderHandle Handle to the compute shader.
   * @param tgSize Thread group dimensions for dispatch.
   * @param bindings Vector of compute bindings (handles or data references).
   * @param refDispatchHandle Handle to a reference dispatch for copying
   * settings.
   */
  ComputeDispatch(const ComputeHandle &shaderHandle = {},
                  const ThreadGroupSize &tgSize = {},
                  const std::vector<ComputeBinding> &bindings = {},
                  const ComputeHandle &refDispatchHandle = {});

  /** Deleted copy constructor. */
  ComputeDispatch(const ComputeDispatch &) = delete;

  /** Default move constructor. */
  ComputeDispatch(ComputeDispatch &&) = default;

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

  ThreadGroupSize tgSize_;                ///< Thread group dimensions.
  ComputeHandle shader_;                  ///< Bound shader handle.
  ComputeHandle referenceDispatchHandle_; ///< Reference dispatch handle.
  std::vector<ComputeBinding> bindings_;  ///< All bindings (handles and data).
};

/**
 * Container for managing ComputeDispatch handles.
 * Provides creation, retrieval, and automatic cleanup of dispatch objects.
 */
class DispatchContainer final : public ComputeDataContainer<> {
private:
  friend class ComputeInterface;

  /** Constructs a DispatchContainer with type ID 1. */
  DispatchContainer() : ComputeDataContainer<>(1) {}
  CONTAINER_METHODS_IMPL(ComputeDispatch);
  CONTAINER_DELETE_METHOD_IMPL(ComputeDispatch);

  /**
   * Destroys a dispatch object when its reference count reaches zero.
   * @param id The handle ID of the object to delete.
   */
  void destroy(size_t id) override { deleteHandlePtr(id); }
};

/**
 * Container for managing DispatchList handles.
 * Provides creation, retrieval, and automatic cleanup of dispatch list objects.
 */
class DispatchListContainer final : public ComputeDataContainer<> {
private:
  friend class ComputeInterface;

  /** Constructs a DispatchListContainer with type ID 2. */
  DispatchListContainer() : ComputeDataContainer<>(2) {}
  CONTAINER_METHODS_IMPL(DispatchList);
  CONTAINER_DELETE_METHOD_IMPL(DispatchList);

  /**
   * Destroys a dispatch list object when its reference count reaches zero.
   * @param id The handle ID of the object to delete.
   */
  void destroy(size_t id) override { deleteHandlePtr(id); }
};

} // namespace cut
