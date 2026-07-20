#include <ComputeInterface.h>

#include <cstring>

namespace cut {

void ComputeInterface::setProfilingEnabled(bool enabled) {
  profilingEnabled_ = enabled;
}

void ComputeInterface::encode(ComputeDispatch &&dispatch) {
  if (!activeCommandBuffer_) {
    activeCommandBuffer_ = commandBufferContainer_->createCommandBuffer();
    auto *cb = commandBufferContainer_->get(activeCommandBuffer_);
    cb->begin();
    cb->setProfilingEnabled(profilingEnabled_);
  }

  commandBufferContainer_->get(activeCommandBuffer_)
      ->encode(std::move(dispatch));
}

ComputeHandle ComputeInterface::submit() {
  flushTransfers();
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

ComputeHandle ComputeInterface::submitReusable() {
  flushTransfers();
  if (!activeCommandBuffer_) {
    logErr("No command buffer is currently recording. "
           "Call encode() before submitReusable().");
  }
  auto *cb = commandBufferContainer_->get(activeCommandBuffer_);
  cb->setReusable(true);
  cb->end();

  ComputeHandle result = std::move(activeCommandBuffer_);
  activeCommandBuffer_.reset();

  commandBufferContainer_->get(result)->submit();

  return result;
}

void ComputeInterface::resubmit(const ComputeHandle &commandBufferHandle) {
  flushTransfers();
  if (!commandBufferHandle) {
    logErr("Invalid command buffer handle. "
           "Call submitReusable() to get a valid handle.");
  }
  commandBufferContainer_->get(commandBufferHandle)->resubmit();
}

void ComputeInterface::wait(const ComputeHandle &commandBufferHandle) {
  if (!commandBufferHandle) {
    logErr("Invalid command buffer handle. "
           "Call submit() to get a valid handle.");
  }
  auto *cb = commandBufferContainer_->get(commandBufferHandle);
  cb->wait();
  const auto &t = cb->lastTimings();
  if (!t.empty())
    lastTimings_.insert(lastTimings_.end(), t.begin(), t.end());
}

std::vector<DispatchTiming> ComputeInterface::takeLastTimings() {
  return std::move(lastTimings_);
}

void ComputeInterface::setCommandBufferContainer(
    std::unique_ptr<CommandBufferContainer> commandBufferContainer) {
  // Reset active command buffer before destroying the old container
  // to avoid use-after-free when the handle's destructor accesses the container
  if (!commandBufferContainer && commandBufferContainer_) {
    activeCommandBuffer_.reset();
  }
  commandBufferContainer_ = std::move(commandBufferContainer);
}

void ComputeInterface::copyActualToAligned(const void *src,
                                           void *dst,
                                           const ComputeBuffer &buffer,
                                           size_t srcOffset,
                                           size_t dstOffset,
                                           size_t size) {
  if (src == nullptr || dst == nullptr) {
    return;
  }

  const size_t elementSize = dataTypeSize(buffer.getDtype());
  const size_t innerDim = buffer.innerDimSize();
  const size_t alignedInnerDim = (innerDim + 3) & ~static_cast<size_t>(3);
  const size_t actualSize = buffer.calculateActualSize();

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

  // Full copy with padding - copy row by row, zeroing padding
  const size_t numRows = buffer.executionSize() / alignedInnerDim;
  const size_t srcRowBytes = innerDim * elementSize;
  const size_t dstRowBytes = alignedInnerDim * elementSize;
  const size_t padBytes = dstRowBytes - srcRowBytes;

  for (size_t row = 0; row < numRows; ++row) {
    std::memcpy(dstBytes + row * dstRowBytes, srcBytes + row * srcRowBytes,
                srcRowBytes);
    // Zero padding elements so vectorized reads (vec4) get correct values
    std::memset(dstBytes + row * dstRowBytes + srcRowBytes, 0, padBytes);
  }
}

void ComputeInterface::copyAlignedToActual(const void *src,
                                           void *dst,
                                           const ComputeBuffer &buffer,
                                           size_t srcOffset,
                                           size_t dstOffset,
                                           size_t size) {
  if (src == nullptr || dst == nullptr) {
    return;
  }

  const size_t elementSize = dataTypeSize(buffer.getDtype());
  const size_t innerDim = buffer.innerDimSize();
  const size_t alignedInnerDim = (innerDim + 3) & ~static_cast<size_t>(3);
  const size_t actualSize = buffer.calculateActualSize();

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
  const size_t numRows = buffer.executionSize() / alignedInnerDim;
  const size_t srcRowBytes = alignedInnerDim * elementSize;
  const size_t dstRowBytes = innerDim * elementSize;

  for (size_t row = 0; row < numRows; ++row) {
    std::memcpy(dstBytes + row * dstRowBytes, srcBytes + row * srcRowBytes,
                dstRowBytes);
  }
}

} // namespace cut
