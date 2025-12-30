#include <ComputeStructs.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace cut {

void ComputeBuffer::setShape(const std::vector<uint32_t> &newShape) {
  if (newShape.size() > 4) {
    throw std::runtime_error("Shape size must be <= 4, got " +
                             std::to_string(newShape.size()));
  }

  // Prepend 1s to ensure size is exactly 4, keeping innermost dimension at back
  const size_t padCount = 4 - newShape.size();
  shape_.assign(padCount, 1);
  shape_.insert(shape_.end(), newShape.begin(), newShape.end());

  // Calculate execution size with overflow checking
  // Innermost dimension (shape_.back()) is aligned to multiple of 4
  executionElementCount_ =
      ((static_cast<size_t>(shape_.back()) + 3) & ~size_t{3});
  size_ = shape_.back();
  for (auto it = shape_.rbegin() + 1; it != shape_.rend(); ++it) {
    executionElementCount_ *= *it;
    size_ *= *it;
  }
  size_ *= dataTypeSize(dtype);
}

std::vector<uint32_t> ComputeBuffer::getDimData() const {
  if (shape_.empty()) {
    throw std::runtime_error("Shape cannot be empty!");
  }
  // Initialize ret as reverse of shape_
  std::vector<uint32_t> ret(shape_.rbegin(), shape_.rend());
  for (size_t i = 2; i < ret.size(); i++) {
    ret[i] = ret[i] * ret[i - 1];
  }
  return ret;
}

size_t ComputeBuffer::executionSize() const {
  return executionElementCount_;
}

size_t ComputeBuffer::calculateActualSize(const std::vector<uint32_t> &shape,
                                          DataType dtype) {
  if (shape.empty()) {
    return 0;
  }
  size_t totalElements = 1;
  for (uint32_t dim : shape) {
    totalElements *= dim;
  }
  return totalElements * dataTypeSize(dtype);
}

size_t ComputeBuffer::calculateAlignedSize(const std::vector<uint32_t> &shape,
                                           DataType dtype) {
  if (shape.empty()) {
    return 0;
  }
  // Round innermost dimension to multiple of 4
  size_t totalElements = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    totalElements *= shape[i];
  }
  size_t alignedInner = (shape.back() + 3) & ~static_cast<uint32_t>(3);
  totalElements *= alignedInner;
  return totalElements * dataTypeSize(dtype);
}

ComputeDispatch::ComputeDispatch(const ComputeHandle &shader,
                                 const ThreadSize &wgSize,
                                 const std::vector<ComputeBinding> &bindings)
    : shader_(shader), wgSize_(wgSize), bindings_(bindings) {}

void ComputeDispatch::bindShader(const ComputeHandle &shaderHandle) {
  shader_ = shaderHandle;
}

void ComputeDispatch::bindResource(const ComputeHandle &resourceHandle,
                                   uint32_t index) {
  bindings_.emplace_back(static_cast<int32_t>(index), resourceHandle);
}

void ComputeDispatch::bindData(const DataReference &data, uint32_t index) {
  if (data.ptr == nullptr) {
    return;
  }

  bindings_.emplace_back(static_cast<int32_t>(index), data);
}

void ComputeDispatch::setWorkgroupSize(const ThreadSize &wgSize) {
  wgSize_ = wgSize;
}

const ThreadSize &ComputeDispatch::workgroupSize() const {
  return wgSize_;
}

const ComputeHandle &ComputeDispatch::shader() const {
  return shader_;
}

const std::vector<ComputeBinding> &ComputeDispatch::bindings() const {
  return bindings_;
}

void ComputeDispatch::sortBindings() {
  std::sort(bindings_.begin(), bindings_.end(),
            [](const ComputeBinding &a, const ComputeBinding &b) {
              return a.index() < b.index();
            });
}

void CommandBuffer::encode(ComputeDispatch &&dispatch) {
  dispatch.sortBindings();
  dispatches_.emplace_back(std::move(dispatch));
}

} // namespace cut
