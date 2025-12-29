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

  // Copy the shape and pad with 1s to ensure size is exactly 4
  shape_ = newShape;
  shape_.resize(4, 1);

  // Calculate execution size with overflow checking
  executionSize_ = ((static_cast<size_t>(shape_.back()) + 3) & ~size_t{3});
  for (auto it = shape_.rbegin() + 1; it != shape_.rend(); ++it) {
    executionSize_ *= *it;
  }
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

size_t ComputeBuffer::getExecutionSize() const {
  return executionSize_;
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
