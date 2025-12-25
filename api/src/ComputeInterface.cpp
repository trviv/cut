#include <ComputeInterface.h>

#include <cstring>

namespace cut {

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

size_t ComputeInterface::calculateActualSize(const std::vector<size_t> &shape,
                                             DataType dtype) {
  if (shape.empty()) {
    return 0;
  }
  size_t totalElements = 1;
  for (size_t dim : shape) {
    totalElements *= dim;
  }
  return totalElements * dataTypeSize(dtype);
}

size_t ComputeInterface::calculateAlignedSize(const std::vector<size_t> &shape,
                                              DataType dtype) {
  if (shape.empty()) {
    return 0;
  }
  // Round innermost dimension to multiple of 4
  std::vector<size_t> alignedShape = shape;
  alignedShape.back() = (alignedShape.back() + 3) & ~static_cast<size_t>(3);

  size_t totalElements = 1;
  for (size_t dim : alignedShape) {
    totalElements *= dim;
  }
  return totalElements * dataTypeSize(dtype);
}

void ComputeInterface::copyActualToAligned(const void *src,
                                           void *dst,
                                           const std::vector<size_t> &shape,
                                           DataType dtype,
                                           size_t srcOffset,
                                           size_t dstOffset) {
  if (shape.empty() || src == nullptr || dst == nullptr) {
    return;
  }

  const size_t elementSize = dataTypeSize(dtype);
  const size_t innerDim = shape.back();
  const size_t alignedInnerDim = (innerDim + 3) & ~static_cast<size_t>(3);

  const auto *srcBytes = static_cast<const char *>(src) + srcOffset;
  auto *dstBytes = static_cast<char *>(dst) + dstOffset;

  // If no padding needed, do a single memcpy
  if (innerDim == alignedInnerDim) {
    std::memcpy(dstBytes, srcBytes, calculateActualSize(shape, dtype));
    return;
  }

  // Calculate number of rows (product of all dimensions except innermost)
  size_t numRows = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    numRows *= shape[i];
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
                                           const std::vector<size_t> &shape,
                                           DataType dtype,
                                           size_t srcOffset,
                                           size_t dstOffset) {
  if (shape.empty() || src == nullptr || dst == nullptr) {
    return;
  }

  const size_t elementSize = dataTypeSize(dtype);
  const size_t innerDim = shape.back();
  const size_t alignedInnerDim = (innerDim + 3) & ~static_cast<size_t>(3);

  const auto *srcBytes = static_cast<const char *>(src) + srcOffset;
  auto *dstBytes = static_cast<char *>(dst) + dstOffset;

  // If no padding needed, do a single memcpy
  if (innerDim == alignedInnerDim) {
    std::memcpy(dstBytes, srcBytes, calculateActualSize(shape, dtype));
    return;
  }

  // Calculate number of rows (product of all dimensions except innermost)
  size_t numRows = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    numRows *= shape[i];
  }

  const size_t srcRowBytes = alignedInnerDim * elementSize;
  const size_t dstRowBytes = innerDim * elementSize;

  for (size_t row = 0; row < numRows; ++row) {
    std::memcpy(dstBytes + row * dstRowBytes, srcBytes + row * srcRowBytes,
                dstRowBytes);
  }
}

} // namespace cut
