#include <ComputeStructs.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace cut {

// --- ComputeData ---

ComputeData::ComputeData(const ComputeHandle &handleRef)
    : type(Type::Handle), handle(handleRef) {}

ComputeData::ComputeData(const DataReference &dataRef) {
  if (dataRef.size <= sizeof(Scalar)) {
    type = Type::Scalar;
    scalar = {};
    std::memcpy(&scalar, dataRef.ptr, dataRef.size);
  } else {
    type = Type::Data;
    const auto *bytePtr = static_cast<const uint8_t *>(dataRef.ptr);
    data = {bytePtr, bytePtr + dataRef.size};
  }
}

const ComputeHandle &ComputeData::getHandle() const {
  return handle;
}

const std::vector<uint8_t> &ComputeData::getData() const {
  if (type != Type::Data) {
    logErr("getData() called on non-Data ComputeData");
  }
  return data;
}

void ComputeBuffer::setShape(const std::vector<uint32_t> &newShape) {
  if (newShape.empty()) {
    throw std::runtime_error("Shape must not be empty");
  }
  if (newShape.size() > 4) {
    throw std::runtime_error("Shape size must be <= 4, got " +
                             std::to_string(newShape.size()));
  }

  shape_ = newShape;

  executionElementCount_ = calculateAlignedElements(shape_);

  // Calculate aligned size
  size_ = calculateAlignedSize(shape_, dtype_);
}

void ComputeBuffer::setDtype(DataType newDtype) {
  dtype_ = newDtype;
  // Recalculate size if shape is already set
  size_ = calculateAlignedSize(shape_, dtype_);
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

uint32_t ComputeBuffer::innerDimSize() const {
  return shape_.empty() ? 0 : shape_.back();
}

size_t ComputeBuffer::calculateActualSize() const {
  return calculateActualSize(shape_, dtype_);
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

size_t
ComputeBuffer::calculateAlignedElements(const std::vector<uint32_t> &shape) {
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
  return totalElements;
}

size_t ComputeBuffer::calculateAlignedSize(const std::vector<uint32_t> &shape,
                                           DataType dtype) {
  const size_t size = calculateAlignedElements(shape) * dataTypeSize(dtype);

  // Align total size to 16 bytes for optimal GPU access
  constexpr size_t kAlignment = 16;
  return (size + kAlignment - 1) & ~(kAlignment - 1);
}

ComputeDispatch::ComputeDispatch(const ComputeHandle &shader,
                                 const ThreadSize &wgSize,
                                 const std::vector<ComputeBinding> &bindings)
    : shader_(shader), wgSize_(wgSize), bindings_(bindings) {}

ComputeDispatch ComputeDispatch::createBarrier() {
  ComputeDispatch barrier;
  barrier.isBarrier_ = true;
  return barrier;
}

ComputeDispatch ComputeDispatch::createBufferUpdate(const ComputeHandle &target,
                                                    const void *data,
                                                    size_t size) {
  ComputeDispatch update;
  update.isBufferUpdate_ = true;
  update.outputHandle_ = target;
  update.bufferUpdateData_.assign(static_cast<const uint8_t *>(data),
                                  static_cast<const uint8_t *>(data) + size);
  // Pad to multiple of 4 bytes (vkCmdUpdateBuffer requirement)
  while (update.bufferUpdateData_.size() % 4 != 0)
    update.bufferUpdateData_.push_back(0);
  return update;
}

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
