#include <ComputeInterface.h>

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

} // namespace cut
