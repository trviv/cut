#include <ComputeStructs.h>

#include <algorithm>
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
}

std::vector<uint32_t> ComputeBuffer::getDimData() const {
  if (shape_.empty()) {
    throw std::runtime_error("Shape cannot be empty!");
  }
  std::vector<uint32_t> ret(4, 1);
  ret[0] = shape_[0];
  if (shape_.size() >= 2) {
    ret[1] = shape_[1];
  }
  for (size_t i = 2; i < shape_.size(); i++) {
    ret[i] = ret[i - 1] * shape_[i];
  }
  for (size_t i = shape_.size(); i < 4; i++) {
    ret[i] = ret[i - 1];
  }
  return ret;
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
