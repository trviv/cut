#pragma once

#include <ComputeHandle.h>
#include <ComputeInterface.h>

#include <cstdint>
#include <vector>

namespace cut {

/**
 * Thin wrapper over ComputeInterface for tensor buffer operations.
 * Provides tensor creation and metadata queries without exposing
 * the full Runtime or GPU backend.
 */
class TensorStore {
public:
  explicit TensorStore(ComputeInterface *iface) : iface_(iface) {}

  const ComputeBuffer &getTensor(const Tensor &handle) const {
    return iface_->getBuffer(handle);
  }

  Tensor createTensor(const std::vector<uint32_t> &shape,
                      DataType dtype,
                      const void *srcPtr = nullptr,
                      bool isUniform = false) {
    return iface_->createBuffer(shape, dtype, srcPtr, isUniform);
  }

  Tensor createTensorEmpty(const std::vector<uint32_t> &shape,
                           DataType dtype,
                           bool isUniform = false) {
    return iface_->createBuffer(shape, dtype, nullptr, isUniform);
  }

private:
  ComputeInterface *iface_;
};

} // namespace cut
