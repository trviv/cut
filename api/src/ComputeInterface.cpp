#include <ComputeInterface.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace cut {

namespace {
// Returns the effective shape by stripping trailing 1s (except the first dim)
// This handles padded shapes like {4,1,1,1} -> {4}
std::vector<uint32_t> getEffectiveShape(const std::vector<uint32_t> &shape) {
  if (shape.empty()) {
    return shape;
  }
  size_t effectiveSize = shape.size();
  while (effectiveSize > 1 && shape[effectiveSize - 1] == 1) {
    --effectiveSize;
  }
  return std::vector<uint32_t>(shape.begin(), shape.begin() + effectiveSize);
}
} // namespace

void ComputeInterface::encode(ComputeDispatch &&dispatch) {
  if (!activeCommandBuffer_) {
    activeCommandBuffer_ = commandBufferContainer_->createCommandBuffer();
    commandBufferContainer_->get(activeCommandBuffer_)->begin();
  }

  commandBufferContainer_->get(activeCommandBuffer_)
      ->encode(std::move(dispatch));
}

ComputeHandle ComputeInterface::submit() {
  if (!activeCommandBuffer_) {
    logErr("No command buffer is currently recording. "
           "Call encode() before submit().");
  }
  commandBufferContainer_->get(activeCommandBuffer_)->end();

  ComputeHandle result = std::move(activeCommandBuffer_);
  activeCommandBuffer_.reset();

  commandBufferContainer_->get(result)->submit();

  return result;
}

void ComputeInterface::wait(const ComputeHandle &commandBufferHandle) {
  if (!commandBufferHandle) {
    logErr("Invalid command buffer handle. "
           "Call submit() to get a valid handle.");
  }
  commandBufferContainer_->get(commandBufferHandle)->wait();
}

void ComputeInterface::setCommandBufferContainer(
    std::unique_ptr<CommandBufferContainer> commandBufferContainer) {
  commandBufferContainer_ = std::move(commandBufferContainer);
}

void ComputeInterface::copyActualToAligned(const void *src,
                                           void *dst,
                                           const std::vector<uint32_t> &shape,
                                           DataType dtype,
                                           size_t srcOffset,
                                           size_t dstOffset,
                                           size_t size) {
  if (shape.empty() || src == nullptr || dst == nullptr) {
    return;
  }

  // Use effective shape (strip trailing 1s from padded shapes)
  const auto effShape = getEffectiveShape(shape);

  const size_t elementSize = dataTypeSize(dtype);
  const size_t innerDim = effShape.back();
  const size_t alignedInnerDim = (innerDim + 3) & ~static_cast<size_t>(3);
  const size_t actualSize = ComputeBuffer::calculateActualSize(effShape, dtype);

  // If size is 0, copy the full buffer
  if (size == 0) {
    size = actualSize;
  }

  const auto *srcBytes = static_cast<const char *>(src) + srcOffset;
  auto *dstBytes = static_cast<char *>(dst) + dstOffset;

  // If no padding needed or this is a partial copy, do a simple memcpy
  if (innerDim == alignedInnerDim ||
      (srcOffset != 0 || dstOffset != 0 || size != actualSize)) {
    std::memcpy(dstBytes, srcBytes, size);
    return;
  }

  // Full copy with padding - copy row by row
  size_t numRows = 1;
  for (size_t i = 0; i < effShape.size() - 1; ++i) {
    numRows *= effShape[i];
  }

  const size_t srcRowBytes = innerDim * elementSize;
  const size_t dstRowBytes = alignedInnerDim * elementSize;

  for (size_t row = 0; row < numRows; ++row) {
    std::memcpy(dstBytes + row * dstRowBytes, srcBytes + row * srcRowBytes,
                srcRowBytes);
  }
}

void ComputeInterface::copyAlignedToActual(const void *src,
                                           void *dst,
                                           const std::vector<uint32_t> &shape,
                                           DataType dtype,
                                           size_t srcOffset,
                                           size_t dstOffset,
                                           size_t size) {
  if (shape.empty() || src == nullptr || dst == nullptr) {
    return;
  }

  // Use effective shape (strip trailing 1s from padded shapes)
  const auto effShape = getEffectiveShape(shape);

  const size_t elementSize = dataTypeSize(dtype);
  const size_t innerDim = effShape.back();
  const size_t alignedInnerDim = (innerDim + 3) & ~static_cast<size_t>(3);
  const size_t actualSize = ComputeBuffer::calculateActualSize(effShape, dtype);

  // If size is 0, copy the full buffer
  if (size == 0) {
    size = actualSize;
  }

  const auto *srcBytes = static_cast<const char *>(src) + srcOffset;
  auto *dstBytes = static_cast<char *>(dst) + dstOffset;

  // If no padding needed or this is a partial copy, do a simple memcpy
  if (innerDim == alignedInnerDim ||
      (srcOffset != 0 || dstOffset != 0 || size != actualSize)) {
    std::memcpy(dstBytes, srcBytes, size);
    return;
  }

  // Full copy with padding - copy row by row
  size_t numRows = 1;
  for (size_t i = 0; i < effShape.size() - 1; ++i) {
    numRows *= effShape[i];
  }

  const size_t srcRowBytes = alignedInnerDim * elementSize;
  const size_t dstRowBytes = innerDim * elementSize;

  for (size_t row = 0; row < numRows; ++row) {
    std::memcpy(dstBytes + row * dstRowBytes, srcBytes + row * srcRowBytes,
                dstRowBytes);
  }
}

} // namespace cut
