#include "TensorStore.h"

namespace cut {

const ComputeBuffer &TensorStore::getTensor(const Tensor &handle) const {
  return iface_->getBuffer(handle);
}

Tensor TensorStore::createTensor(const std::vector<uint32_t> &shape,
                                 DataType dtype,
                                 const void *srcPtr,
                                 bool isUniform) {
  return iface_->createBuffer(shape, dtype, srcPtr, isUniform);
}

Tensor TensorStore::createTensorEmpty(const std::vector<uint32_t> &shape,
                                      DataType dtype,
                                      bool isUniform) {
  return iface_->createBuffer(shape, dtype, nullptr, isUniform);
}

Tensor TensorStore::acquireTempBuffer(size_t numElements, DataType dtype) {
  size_t sizeBytes = ComputeBuffer::calculateAlignedSize(
      {static_cast<uint32_t>(numElements)}, dtype);

  for (auto it = tempBufferPool_.begin(); it != tempBufferPool_.end(); ++it) {
    const auto &buffer = iface_->getBuffer(*it);
    if (buffer.size() >= sizeBytes) {
      Tensor handle = *it;
      tempBufferPool_.erase(it);
      activeTempBuffers_.push_back(handle);
      return handle;
    }
  }

  Tensor handle =
      iface_->createBuffer({static_cast<uint32_t>(numElements)}, dtype);
  activeTempBuffers_.push_back(handle);
  return handle;
}

void TensorStore::releaseTempBuffers() {
  for (const auto &handle : activeTempBuffers_) {
    tempBufferPool_.push_back(handle);
  }
  activeTempBuffers_.clear();
}

} // namespace cut
