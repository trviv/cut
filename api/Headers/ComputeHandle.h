#pragma once

#include <cstddef>
#include <cstdint>

namespace cut {

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
  friend class ComputeContainer;

  /**
   * Private constructor used by ComputeContainer to create valid handles.
   * @param container The container that owns the referenced object.
   * @param handleId The unique identifier for the object within the container.
   */
  ComputeHandle(ComputeContainer *container, size_t handleId);

  size_t id_; ///< Handle ID within its container.
  ComputeContainer *container_; ///< Compute object container the handle belongs to.
};

} // namespace cut
