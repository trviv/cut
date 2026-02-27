#pragma once

#include <ComputeHandle.h>
#include <ComputeInterface.h>

#include <cstdint>
#include <vector>

namespace cut {

/**
 * Thin wrapper over ComputeInterface for tensor buffer operations.
 * Provides tensor creation, metadata queries, and temporary buffer
 * pooling for multi-pass operations.
 */
class TensorStore {
public:
  explicit TensorStore(ComputeInterface *iface) : iface_(iface) {}

  /// Returns the underlying compute interface (used by Dispatcher for
  /// encode/shader operations).
  ComputeInterface *iface() const { return iface_; }

  const ComputeBuffer &getTensor(const Tensor &handle) const;

  Tensor createTensor(const std::vector<uint32_t> &shape,
                      DataType dtype,
                      const void *srcPtr = nullptr,
                      bool isUniform = false);

  Tensor createTensorEmpty(const std::vector<uint32_t> &shape,
                           DataType dtype,
                           bool isUniform = false);

  /// Creates a tensor view referencing a sub-region of an existing tensor.
  /// The view shares the parent's GPU buffer at the given byte offset.
  Tensor createTensorView(const Tensor &parent,
                          size_t byteOffset,
                          const std::vector<uint32_t> &shape,
                          DataType dtype);

  /// Acquires a temporary GPU buffer from the pool or creates a new one.
  /// Used by multi-pass OpNodes to allocate intermediate buffers.
  Tensor acquireTempBuffer(size_t numElements, DataType dtype);

  /// Releases all temporary buffers back to the pool.
  void releaseTempBuffers();

private:
  ComputeInterface *iface_;

  /// Pool of reusable temporary GPU buffers.
  std::vector<Tensor> tempBufferPool_;

  /// Temporary buffers currently in use by the current multi-pass operation.
  std::vector<Tensor> activeTempBuffers_;
};

} // namespace cut
