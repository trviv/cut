#pragma once

#include <ComputeHandle.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cut {

// Forward declaration for logging function
extern void logErr(const char *format, ...);

/**
 * Base container class for managing compute objects such as buffers,
 * textures, shaders, etc.
 *
 * Handles reference counting automatically - objects are deleted when
 * their reference count reaches zero.
 */
class ComputeContainer {
protected:
  /**
   * Type-erased storage for handle data.
   * Stores arbitrary data up to sizeof(void*) bytes.
   */
  class HandleData final {
  public:
    /**
     * Constructs HandleData from any type that fits within void*.
     * @tparam T The type of data to store.
     * @param data The data value to store.
     */
    template <typename T>
    HandleData(const T data) {
      static_assert(sizeof(T) <= sizeof(void *),
                    "ComputeContainer can't store data larger than: "
                    "sizeof(void*)");
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
    void *data_; ///< Raw storage for the handle data.
  };

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
   * @param data The handle data for the object to destroy.
   */
  virtual void destroy(HandleData &data) = 0;

  /**
   * Retrieves the data associated with a handle.
   * @param handle The handle to look up.
   * @return Const reference to the handle's data.
   */
  const HandleData &data(const ComputeHandle &handle) const;

  /**
   * Creates a new handle for the given data.
   * @param data The data to associate with the new handle.
   * @return A new ComputeHandle referencing the data.
   */
  ComputeHandle createHandle(const HandleData &data);

  /**
   * Verifies that a handle is valid and belongs to this container.
   * @param handle The handle to verify.
   */
  void verify(const ComputeHandle &handle) const;

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

  std::vector<HandleData> objects_; ///< Storage for all managed object data.
  std::vector<size_t> refCounts_;   ///< Reference counts for each object.
  std::vector<size_t> freeHandles_; ///< Pool of reusable handle IDs.
  const uint32_t type_; ///< Unique type identifier for this container.
};

} // namespace cut
