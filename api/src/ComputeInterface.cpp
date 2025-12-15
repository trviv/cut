#include <ComputeInterface.h>

namespace cut {

void ComputeInterface::beginCommandBuffer() {
  if (activeCommandBuffer_) {
    logErr("Cannot begin a new command buffer while one is already recording. "
           "Call endCommandBuffer() first.");
  }
  activeCommandBuffer_ = createCommandBuffer();
}

ComputeHandle ComputeInterface::endCommandBuffer() {
  if (!activeCommandBuffer_) {
    logErr("No command buffer is currently recording. "
           "Call beginCommandBuffer() first.");
  }
  ComputeHandle result =
      commandBufferContainer_.create(std::move(activeCommandBuffer_));
  activeCommandBuffer_.reset();
  return result;
}

ComputeHandle ComputeInterface::encode(ComputeDispatch &&dispatch) {
  if (!activeCommandBuffer_) {
    logErr("No command buffer is currently recording. "
           "Call beginCommandBuffer() before encode().");
  }

  activeCommandBuffer_->encode(std::move(dispatch));
  return {}; // Return empty handle as dispatch is now owned by command buffer
}

const CommandBuffer &
ComputeInterface::getCommandBuffer(const ComputeHandle &handle) const {
  return *commandBufferContainer_.get(handle);
}

} // namespace cut
