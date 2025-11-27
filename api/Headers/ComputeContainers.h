#pragma once

#include <ComputeHandle.h>

namespace cut {

/**
 * Represents a binding for a compute dispatch operation.
 * Can hold either a ComputeHandle (for buffers/textures) or a DataReference
 * (for push constants).
 */
struct ComputeBinding final {
  /**
   * Indicates whether the binding contains a ComputeHandle or DataReference.
   */
  enum class Type { Handle, Data };

  /**
   * Constructs a ComputeBinding with a ComputeHandle.
   * @param index The binding index in the shader.
   * @param handle The ComputeHandle to bind.
   */
  ComputeBinding(int32_t index, const ComputeHandle &handle)
      : bindingIndex(index), type(Type::Handle), handle(handle) {}

  /**
   * Constructs a ComputeBinding with a DataReference.
   * @param index The binding index in the shader.
   * @param data The DataReference to bind.
   */
  ComputeBinding(int32_t index, const DataReference &data)
      : bindingIndex(index), type(Type::Data), dataRef(data) {}

  /** Destructor. */
  ~ComputeBinding() {
    if (type == Type::Handle) {
      handle.~ComputeHandle();
    }
  }

  /** Copy constructor. */
  ComputeBinding(const ComputeBinding &other)
      : bindingIndex(other.bindingIndex), type(other.type) {
    if (type == Type::Handle) {
      new (&handle) ComputeHandle(other.handle);
    } else {
      new (&dataRef) DataReference(other.dataRef);
    }
  }

  /** Move constructor. */
  ComputeBinding(ComputeBinding &&other)
      : bindingIndex(other.bindingIndex), type(other.type) {
    if (type == Type::Handle) {
      new (&handle) ComputeHandle(std::move(other.handle));
    } else {
      new (&dataRef) DataReference(other.dataRef);
    }
  }

  /**
   * Checks if this binding contains a ComputeHandle.
   * @return True if the binding holds a ComputeHandle.
   */
  bool isHandle() const { return type == Type::Handle; }

  /**
   * Checks if this binding contains a DataReference.
   * @return True if the binding holds a DataReference.
   */
  bool isData() const { return type == Type::Data; }

  union {
    ComputeHandle handle;  ///< Handle to a buffer or texture.
    DataReference dataRef; ///< Reference to raw data (push constants).
  };

  int32_t bindingIndex; ///< The binding index in the shader.
  Type type;            ///< Indicates which union member is active.
};

/** Forward declarations. */
class ComputeInterface;

/**
 * @def CONTAINER_METHODS_IMPL(STRUCT_NAME)
 * Macro that generates standard container methods for a given struct type.
 * Generates: createHandle, get (mutable/const from handle), get (mutable/const
 * from data).
 */
#define CONTAINER_METHODS_IMPL(STRUCT_NAME)                                    \
  ComputeHandle createHandle(STRUCT_NAME &&structData) {                       \
    return ComputeContainer::createHandle(                                     \
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
 * Macro that generates a deletion method for handle data.
 * Safely deletes the pointer stored in handle data if non-null.
 */
#define CONTAINER_DELETE_METHOD_IMPL(STRUCT_NAME)                              \
  void deleteHandlePtr(HandleData &handleData) {                               \
    auto *ptr = get(handleData);                                               \
    if (ptr != nullptr) {                                                      \
      delete ptr;                                                              \
    }                                                                          \
  }

/**
 * A list of dispatch handles to be executed sequentially.
 */
using DispatchList = std::vector<ComputeHandle>;

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
   * @param refDispatchHandle Handle to a reference dispatch for copying
   * settings.
   * @param resourceBindings Vector of buffer/texture handles to bind.
   * @param dataBindings Vector of data references to bind.
   */
  ComputeDispatch(const ComputeHandle &shaderHandle = {},
                  const ThreadGroupSize &tgSize = {},
                  const ComputeHandle &refDispatchHandle = {},
                  const std::vector<ComputeHandle> &resourceBindings = {},
                  const std::vector<DataReference> &dataBindings = {});

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

  ThreadGroupSize tgSize_;                      ///< Thread group dimensions.
  ComputeHandle shader_;                        ///< Bound shader handle.
  ComputeHandle referenceDispatchHandle_;       ///< Reference dispatch handle.
  std::vector<ComputeHandle> resourceBindings_; ///< Bound resources.
  std::vector<std::vector<uint8_t>>
      dataBindings_; ///< Bound data (push constants).
};

/**
 * Container for managing ComputeDispatch handles.
 * Provides creation, retrieval, and automatic cleanup of dispatch objects.
 */
class DispatchContainer final : public ComputeContainer {
private:
  friend class ComputeInterface;

  /** Constructs a DispatchContainer with type ID 1. */
  DispatchContainer() : ComputeContainer(1) {}
  CONTAINER_METHODS_IMPL(ComputeDispatch);
  CONTAINER_DELETE_METHOD_IMPL(ComputeDispatch);

  /**
   * Destroys a dispatch object when its reference count reaches zero.
   * @param data The handle data containing the pointer to delete.
   */
  void destroy(HandleData &data) override { deleteHandlePtr(data); }
};

/**
 * Container for managing DispatchList handles.
 * Provides creation, retrieval, and automatic cleanup of dispatch list objects.
 */
class DispatchListContainer final : public ComputeContainer {
private:
  friend class ComputeInterface;

  /** Constructs a DispatchListContainer with type ID 2. */
  DispatchListContainer() : ComputeContainer(2) {}
  CONTAINER_METHODS_IMPL(DispatchList);
  CONTAINER_DELETE_METHOD_IMPL(ComputeDispatch);

  /**
   * Destroys a dispatch list object when its reference count reaches zero.
   * @param data The handle data containing the pointer to delete.
   */
  void destroy(HandleData &data) override { deleteHandlePtr(data); }
};

} // namespace cut
