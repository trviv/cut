#include <ComputeInterface.h>

namespace cut {

void ComputeInterface::beginCommandBuffer() {
  if (activeCommandBuffer_) {
    logErr("Cannot begin a new command buffer while one is already recording. "
           "Call endCommandBuffer() first.");
  }
  activeCommandBuffer_ = commandBufferContainer_->createCommandBuffer();
  commandBufferContainer_->get(activeCommandBuffer_)->begin();
}

void ComputeInterface::encode(ComputeDispatch &&dispatch) {
  if (!activeCommandBuffer_) {
    logErr("No command buffer is currently recording. "
           "Call beginCommandBuffer() before encode().");
  }

  commandBufferContainer_->get(activeCommandBuffer_)
      ->encode(std::move(dispatch));
}

ComputeHandle ComputeInterface::endCommandBuffer() {
  if (!activeCommandBuffer_) {
    logErr("No command buffer is currently recording. "
           "Call beginCommandBuffer() first.");
  }
  commandBufferContainer_->get(activeCommandBuffer_)->end();

  ComputeHandle result = std::move(activeCommandBuffer_);
  activeCommandBuffer_.reset();

  return result;
}

void ComputeInterface::submit(const ComputeHandle &commandBufferHandle) {
  if (!commandBufferHandle) {
    logErr("Invalid command buffer handle. "
           "Call endCommandBuffer() to get a valid handle.");
  }
  commandBufferContainer_->get(commandBufferHandle)->submit();
}

void ComputeInterface::setCommandBufferContainer(
    std::unique_ptr<CommandBufferContainer> commandBufferContainer) {
  commandBufferContainer_ = std::move(commandBufferContainer);
}

} // namespace cut
