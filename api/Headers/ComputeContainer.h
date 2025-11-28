#pragma once

#include <ComputeHandle.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cut {

// Forward declaration for logging function
extern void logErr(const char *format, ...);

/**
 * Base container class for managing compute object handles.
 *
 * Handles reference counting automatically - objects are deleted when
 * their reference count reaches zero.
 */
class ComputeContainer {
protected:
  /**
   * Constructs a ComputeContainer with the specified type identifier.
   * @param type A unique identifier for the container type.
   */
  ComputeContainer(uint32_t type);

  /** Virtual destructor. Cleans up any remaining objects. */
  virtual ~ComputeContainer();

  /**
   * Pure virtual function to deallocate API-level objects.
   * Must be implemented by derived classes to perform proper cleanup.
   * Called automatically when an object's reference count reaches zero.
   * @param id The handle ID for the object to destroy.
   */
  virtual void destroy(size_t id) = 0;

  /**
   * Allocates a new handle slot.
   * @return The index of the allocated slot.
   */
  size_t allocateSlot();

  /**
   * Creates a new handle for the given slot index.
   * @param index The slot index to create a handle for.
   * @return A new ComputeHandle referencing the slot.
   */
  ComputeHandle createHandleFromSlot(size_t index);

  /**
   * Verifies that a handle is valid and belongs to this container.
   * @param handle The handle to verify.
   */
  void verify(const ComputeHandle &handle) const;

  /**
   * Returns the number of allocated slots.
   */
  size_t slotCount() const { return refCounts_.size(); }

  /**
   * Returns the number of free handle slots.
   */
  size_t freeSlotCount() const { return freeHandles_.size(); }

private:
  friend class ComputeHandle;

  /**
   * Increments the reference count for a handle.
   * @param handle The handle whose reference count to increment.
   */
  void addRef(const ComputeHandle &ref);

  /**
   * Decrements the reference count for a handle.
   * Destroys the object if the count reaches zero.
   * @param handle The handle whose reference count to decrement.
   */
  void remRef(ComputeHandle &ref);

  std::vector<size_t> refCounts_;   ///< Reference counts for each object.
  std::vector<size_t> freeHandles_; ///< Pool of reusable handle IDs.
  const uint32_t type_; ///< Unique type identifier for this container.
};

/**
 * Template container class that adds data storage to ComputeContainer.
 *
 * Provides type-erased storage for handle data and associated methods.
 * @tparam DataType The underlying storage type for handle data (default: void*)
 */
template <typename DataType>
class ComputeDataContainer : public ComputeContainer {
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
                    "ComputeDataContainer can't store data larger than: "
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

protected:
  /**
   * Constructs a ComputeDataContainer with the specified type identifier.
   * @param type A unique identifier for the container type.
   */
  ComputeDataContainer(uint32_t type) : ComputeContainer(type) {}

  /** Virtual destructor. */
  virtual ~ComputeDataContainer() {
    if (objects_.size() != freeSlotCount()) {
      logErr("Trying to destroy container before all objects in it have "
             "been deallocated.");
    }
    objects_.clear();
  }

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
    return objects_[handle.id_];
  }

  /**
   * Creates a new handle for the given data.
   * @param data The data to associate with the new handle.
   * @return A new ComputeHandle referencing the data.
   */
  ComputeHandle createHandle(HandleData &&data) {
    size_t index = allocateSlot();

    if (index < objects_.size()) {
      objects_[index] = std::move(data);
    } else {
      objects_.emplace_back(std::move(data));
    }

    return createHandleFromSlot(index);
  }

  std::vector<HandleData> objects_; ///< Storage for all managed object data.
};

} // namespace cut
