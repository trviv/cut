#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cut {

// Forward declaration for logging function
extern void logErr(const char *format, ...);

template <typename DataType = size_t>
class ComputeContainer;

/**
 * A handle class used to reference compute objects such as buffers,
 * textures, shaders, etc. Provides automatic reference counting.
 */
class ComputeHandle final {
public:
  /** Constructs an empty (invalid) handle. */
  ComputeHandle();

  /** Destructor. Decrements reference count on the associated object. */
  ~ComputeHandle();

  /**
   * Copy constructor. Increments the reference count.
   * @param ref The handle to copy from.
   */
  ComputeHandle(const ComputeHandle &ref);

  /**
   * Move constructor. Transfers ownership without changing reference count.
   * @param ref The handle to move from.
   */
  ComputeHandle(ComputeHandle &&ref);

  /**
   * Boolean conversion operator.
   * @return True if the handle references a valid object, false otherwise.
   */
  operator bool() const;

  /**
   * Copy assignment operator. Releases current reference and acquires new one.
   * @param ref The handle to copy from.
   */
  void operator=(const ComputeHandle &ref);

  /** Releases the reference to the current object, making the handle invalid.
   */
  void reset();

private:
  template <typename DataType>
  friend class ComputeContainer;

  /**
   * Private constructor used by ComputeContainer to create valid handles.
   * @param container The container that owns the referenced object.
   * @param handleId The unique identifier for the object within the container.
   */
  ComputeHandle(ComputeContainer<> *container, size_t handleId);

  size_t id_; ///< Handle ID within its container.
  ComputeContainer<>
      *container_; ///< Compute object container the handle belongs to.
};

/**
 * Base container class for managing compute objects such as buffers,
 * textures, shaders, etc.
 *
 * Handles reference counting automatically - objects are deleted when
 * their reference count reaches zero.
 *
 * @tparam DataType The type used for storing handle data.
 */
template <typename DataType>
class ComputeContainer {
protected:
  /**
   * Type-erased storage for handle data.
   * Stores arbitrary data up to sizeof(DataType) bytes.
   */
  class HandleData final {
  public:
    /**
     * Constructs HandleData from any type that fits within DataType.
     * @tparam T The type of data to store.
     * @param data The data value to store.
     */
    template <typename T>
    HandleData(const T data) {
      static_assert(sizeof(T) <= sizeof(DataType),
                    "ComputeContainer can't store data larger than: "
                    "sizeof(DataType)");
      std::memcpy(&data_, &data, sizeof(T));
    }

    /**
     * Retrieves the stored data as the specified type.
     * @tparam T The type to interpret the stored data as.
     * @return A copy of the stored data interpreted as type T.
     */
    template <typename T>
    T get() const {
      T ret;
      std::memcpy(&ret, &data_, sizeof(T));
      return std::move(ret);
    }

  private:
    DataType data_; ///< Raw storage for the handle data.
  };

  /**
   * Complete handle storage structure containing reference count and data.
   */
  class HandleStruct final {
  public:
    /**
     * Constructs a HandleStruct with the given data and zero reference count.
     * @param data The handle data to store.
     */
    HandleStruct(const HandleData &data) : refCount(0), data(data) {}

  private:
    friend class ComputeContainer;

    size_t refCount; ///< Number of active references to this object.
    HandleData data; ///< The stored handle data.
  };

protected:
  /**
   * Constructs a ComputeContainer with the specified type identifier.
   * @param type A unique identifier for the container type.
   */
  ComputeContainer(uint32_t type) : type_(type) {}

  /** Virtual destructor. Cleans up any remaining objects. */
  virtual ~ComputeContainer() {
    if (objects_.size() != freeHandles_.size()) {
      logErr("Trying to destroy container before all objects in it have "
             "been deallocated.");
    }
    objects_.clear();
    freeHandles_.clear();
  }

  /**
   * Pure virtual function to deallocate API-level objects.
   * Must be implemented by derived classes to perform proper cleanup.
   * Called automatically when an object's reference count reaches zero.
   * @param data The handle data for the object to destroy.
   */
  virtual void destroy(HandleData &data) = 0;

  /**
   * Retrieves the data associated with a handle.
   * @param handle The handle to look up.
   * @return Const reference to the handle's data.
   */
  const HandleData &data(const ComputeHandle &handle) const {
    if (!handle) {
      throw std::runtime_error("Trying to get data for an empty handle");
    }
    verify(handle);

    return objects_[handle.id_].data;
  }

  /**
   * Creates a new handle for the given data.
   * @param data The data to associate with the new handle.
   * @return A new ComputeHandle referencing the data.
   */
  ComputeHandle createHandle(const HandleData &data) {
    size_t index;
    if (!freeHandles_.empty()) {
      index = freeHandles_.back();
      freeHandles_.pop_back();
      objects_[index] = HandleStruct(data);
    } else {
      index = objects_.size();
      objects_.emplace_back(HandleStruct(data));
    }

    return ComputeHandle(reinterpret_cast<ComputeContainer<> *>(this), index);
  }

  /**
   * Verifies that a handle is valid and belongs to this container.
   * @param handle The handle to verify.
   */
  void verify(const ComputeHandle &handle) const {
    if (handle.container_ != reinterpret_cast<const ComputeContainer<> *>(this)) {
      throw std::runtime_error("Trying to get data for handle which does not "
                               "belong to the container");
    }
  }

private:
  friend class ComputeHandle;

  /**
   * Increments the reference count for a handle.
   * @param handle The handle whose reference count to increment.
   */
  void addRef(const ComputeHandle &ref) {
    // Add object reference
    objects_[ref.id_].refCount++;
  }

  /**
   * Decrements the reference count for a handle.
   * Destroys the object if the count reaches zero.
   * @param handle The handle whose reference count to decrement.
   */
  void remRef(ComputeHandle &ref) {
    // Reduce object reference
    auto &objectRef = objects_[ref.id_];
    objectRef.refCount--;
    // If object references reached zero then object can be deallocated
    if (objectRef.refCount == 0) {
      // Deallocate the object based on API specific implementation
      destroy(objectRef.data);
      // Add object handle to free handle list
      freeHandles_.push_back(ref.id_);
      ref.container_ = nullptr;
    }
  }

  std::vector<HandleStruct> objects_; ///< Storage for all managed objects.
  std::vector<size_t> freeHandles_;   ///< Pool of reusable handle IDs.
  const uint32_t type_; ///< Unique type identifier for this container.
};

} // namespace cut
