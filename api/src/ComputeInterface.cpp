#include <ComputeInterface.h>

namespace cut {

void ComputeInterface::beginCommandBuffer() {
  if (activeCommandBuffer_) {
    logErr("Cannot begin a new command buffer while one is already recording. "
           "Call endCommandBuffer() first.");
  }
  activeCommandBuffer_ = commandBufferContainer_.create();
}

ComputeHandle ComputeInterface::endCommandBuffer() {
  if (!activeCommandBuffer_) {
    logErr("No command buffer is currently recording. "
           "Call beginCommandBuffer() first.");
  }
  ComputeHandle result = activeCommandBuffer_;
  activeCommandBuffer_.reset();
  return result;
}

ComputeHandle ComputeInterface::encode(ComputeDispatch &&dispatch) {
  if (!activeCommandBuffer_) {
    logErr("No command buffer is currently recording. "
           "Call beginCommandBuffer() before encode().");
  }

  auto &cmdBuffer = *commandBufferContainer_.get(activeCommandBuffer_);

  if (dispatch.referenceDispatchHandle_) {
    const auto &refDispatch = cmdBuffer.get(dispatch.referenceDispatchHandle_);
    if (refDispatch.referenceDispatchHandle_) {
      logErr("Creating dispatch to a reference which itself is a reference, "
             "is not allowed.");
    }
  }

  return cmdBuffer.encode(std::move(dispatch));
}

CommandBuffer &ComputeInterface::getCommandBuffer(const ComputeHandle &handle) {
  return *commandBufferContainer_.get(handle);
}

const CommandBuffer &
ComputeInterface::getCommandBuffer(const ComputeHandle &handle) const {
  return *commandBufferContainer_.get(handle);
}

ComputeHandle CommandBuffer::encode(ComputeDispatch &&dispatch) {
  return create(std::move(dispatch));
}

} // namespace cut
