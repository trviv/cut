#pragma once

#include <ComputeHandle.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
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
public:
  /**
   * Returns the number of active (in-use) objects in the container.
   */
  size_t size() const { return slotCount() - freeSlotCount(); }

  /**
   * Returns the name identifier for the container type.
   */
  std::string_view name() const { return name_; }

protected:
  /**
   * Constructs a ComputeContainer with the specified name.
   * @param name The name identifier for the container type.
   */
  ComputeContainer(std::string_view name);

  /** Virtual destructor. Cleans up any remaining objects. */
  virtual ~ComputeContainer();

  /**
   * Pure virtual function to deallocate API-level objects.
   * Must be implemented by derived classes to perform proper cleanup.
   * Called automatically when an object's reference count reaches zero.
   * @param handle The handle for the object to destroy.
   */
  virtual void destroy(const ComputeHandle &handle) = 0;

  /**
   * Allocates a new handle slot.
   * @return The index of the allocated slot.
   */
  size_t allocateSlot();

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
  const std::string_view name_;
};

/**
 * Template container class that adds data storage to ComputeContainer.
 *
 * When DataType is a pointer type (e.g., void*), provides type-erased storage
 * via memcpy for small types that fit within sizeof(DataType).
 *
 * When DataType is a non-pointer type, stores DataType directly with proper
 * move semantics.
 *
 * @tparam DataType The underlying storage type for handle data
 */
template <typename DataType>
class ComputeDataContainer : public ComputeContainer {
  static constexpr bool IsPointer = std::is_pointer<DataType>::value;

  /// Helper to enable methods only for pointer types
  template <bool P = IsPointer>
  using EnableIfPointer = typename std::enable_if<P, int>::type;

  /// Helper to enable methods only for non-pointer types
  template <bool P = IsPointer>
  using EnableIfNotPointer = typename std::enable_if<!P, int>::type;

  /// The underlying type (removes pointer if DataType is a pointer)
  using UnderlyingType = typename std::
      conditional<IsPointer, std::remove_pointer_t<DataType>, DataType>::type;

  static_assert(
      std::is_same<std::remove_cv_t<decltype(UnderlyingType::Name)>,
                   std::string_view>::value,
      "DataType (or pointed-to type) must have a static Name member of type "
      "std::string_view");

  friend class ComputeHandle;

public:
  /**
   * Retrieves a const reference to the stored data (non-pointer types).
   * @param handle The handle to get data for.
   * @return Const reference to the stored data.
   */
  template <bool P = IsPointer, EnableIfNotPointer<P> = 0>
  const DataType &get(const ComputeHandle &handle) const {
    if (!handle) {
      throw std::runtime_error("Trying to get data for an empty handle");
    }
    verify(handle);
    return objects_[handle.id_];
  }

protected:
  ComputeDataContainer() : ComputeContainer(UnderlyingType::Name) {}

  virtual ~ComputeDataContainer() {
    if (objects_.size() != freeSlotCount()) {
      logErr("Trying to destroy container before all objects in it have "
             "been deallocated.");
    }
    objects_.clear();
  }

  /**
   * Creates a new handle for the given data.
   * @param hdata The data to store.
   * @return A new ComputeHandle referencing the data.
   */
  ComputeHandle create(DataType &&hdata) {
    size_t index = allocateSlot();

    if (index < objects_.size()) {
      objects_[index] = std::move(hdata);
    } else {
      objects_.emplace_back(std::move(hdata));
    }

    return ComputeHandle(this, index);
  }

  /**
   * Retrieves the stored data for a handle (pointer types).
   * @param handle The handle to get data for.
   * @return The stored pointer value.
   */
  template <bool P = IsPointer, EnableIfPointer<P> = 0>
  DataType get(const ComputeHandle &handle) const {
    if (!handle) {
      throw std::runtime_error("Trying to get data for an empty handle");
    }
    verify(handle);
    return objects_[handle.id_];
  }

  /**
   * Retrieves a mutable reference to the stored data (non-pointer types).
   * @param handle The handle to get data for.
   * @return Mutable reference to the stored data.
   */
  template <bool P = IsPointer, EnableIfNotPointer<P> = 0>
  DataType &get(const ComputeHandle &handle) {
    if (!handle) {
      throw std::runtime_error("Trying to get data for an empty handle");
    }
    verify(handle);
    return objects_[handle.id_];
  }

  /**
   * Destroys API-specific resources associated with the handle.
   * Override this in derived classes to release backend-specific objects
   * (e.g., Vulkan buffers, Metal textures) before the container data is reset.
   * Called automatically by destroy() before the object slot is recycled.
   * @param handle The handle of the object whose API resources should be freed.
   */
  virtual void destroyAPIObject(const ComputeHandle &handle) {}

private:
  /**
   * Default destroy implementation.
   * For pointer types: deletes the pointer and sets it to nullptr.
   * For non-pointer types: resets the object to default state.
   * @param handle The handle of the object to destroy.
   */
  void destroy(const ComputeHandle &handle) override {
    destroyAPIObject(handle);
    destroyImpl(handle.id_);
  }

  /// Destroy implementation for pointer types: delete and nullify
  template <bool P = IsPointer, EnableIfPointer<P> = 0>
  void destroyImpl(size_t id) {
    delete objects_[id];
    objects_[id] = nullptr;
  }

  /// Destroy implementation for non-pointer types: reset to default
  template <bool P = IsPointer, EnableIfNotPointer<P> = 0>
  void destroyImpl(size_t id) {
    objects_[id] = DataType{};
  }

  std::vector<DataType> objects_;
};

// Template implementation for ComputeHandle::get<T>()
template <typename T>
const T &ComputeHandle::get() const {
  if (!*this) {
    throw std::runtime_error("Trying to get data for an empty handle");
  }
  if (container_->name() != T::Name) {
    throw std::runtime_error("Type mismatch: handle container name '" +
                             std::string(container_->name()) +
                             "' does not match requested type '" +
                             std::string(T::Name) + "'");
  }
  auto *dataContainer =
      static_cast<const ComputeDataContainer<T> *>(container_);
  return dataContainer->get(*this);
}

} // namespace cut
