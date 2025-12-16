#include <ComputeInterface.h>

namespace cut {

void ComputeInterface::beginCommandBuffer() {
  if (activeCommandBuffer_) {
    logErr("Cannot begin a new command buffer while one is already recording. "
           "Call endCommandBuffer() first.");
  }
  activeCommandBuffer_ = commandBufferContainer_->createCommandBuffer();
}

ComputeHandle ComputeInterface::endCommandBuffer() {
  if (!activeCommandBuffer_) {
    logErr("No command buffer is currently recording. "
           "Call beginCommandBuffer() first.");
  }
  ComputeHandle result = std::move(activeCommandBuffer_);
  activeCommandBuffer_.reset();
  return result;
}

void ComputeInterface::encode(ComputeDispatch &&dispatch) {
  if (!activeCommandBuffer_) {
    logErr("No command buffer is currently recording. "
           "Call beginCommandBuffer() before encode().");
  }

  commandBufferContainer_->get(activeCommandBuffer_)
      ->encode(std::move(dispatch));
}

const CommandBuffer &
ComputeInterface::getCommandBuffer(const ComputeHandle &handle) const {
  return *commandBufferContainer_->get(handle);
}

CommandBufferContainer &ComputeInterface::getCommandBufferContainer() {
  return *commandBufferContainer_;
}

void ComputeInterface::setCommandBufferContainer(
    std::unique_ptr<CommandBufferContainer> commandBufferContainer) {
  commandBufferContainer_ = std::move(commandBufferContainer);
}

} // namespace cut
