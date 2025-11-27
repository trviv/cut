#include <ComputeInterface.h>

namespace cut {

ComputeHandle
ComputeInterface::createDispatch(const ComputeHandle &shader,
                                 const ThreadGroupSize &tgSize,
                                 std::vector<ComputeHandle> &&resourceBindings,
                                 std::vector<DataReference> &&dataReferences) {
  return dispatchContainer_.createHandle(
      {shader, tgSize, {}, resourceBindings, dataReferences});
}

ComputeHandle
ComputeInterface::createDispatchFromRef(const ComputeHandle &refhandle) {
  if (dispatchContainer_.get(refhandle)->referenceDispatchHandle_) {
    logErr("Creating dispatch to a reference which itself is a reference, "
           "is not allowed.");
  }

  return dispatchContainer_.createHandle({{}, {}, refhandle, {}, {}});
}

void ComputeInterface::bindShader(const ComputeHandle &dispatchHandle,
                                  const ComputeHandle &shaderHandle) {
  dispatchContainer_.get(dispatchHandle)->bindShader(shaderHandle);
}

void ComputeInterface::bindResource(const ComputeHandle &dispatchHandle,
                                    const ComputeHandle &resourceHandle,
                                    uint32_t bindIndex) {
  dispatchContainer_.get(dispatchHandle)
      ->bindResource(resourceHandle, bindIndex);
}

void ComputeInterface::bindResources(
    const ComputeHandle &dispatchHandle,
    const std::vector<ComputeHandle> &resourceHandles, uint32_t bindOffset) {
  auto *dispatch = dispatchContainer_.get(dispatchHandle);
  for (uint32_t i = 0; i < resourceHandles.size(); ++i) {
    dispatch->bindResource(resourceHandles[i], bindOffset + i);
  }
}

void ComputeInterface::bindData(const ComputeHandle &dispatchHandle,
                                const DataReference &data, uint32_t bindIndex) {
  dispatchContainer_.get(dispatchHandle)->bindData(data, bindIndex);
}

void ComputeInterface::setThreadGroupSize(const ComputeHandle &dispatchHandle,
                                          const ThreadGroupSize &tgSize) {
  dispatchContainer_.get(dispatchHandle)->setThreadGroupSize(tgSize);
}

ComputeHandle ComputeInterface::createDispatchList(DispatchList &&dispatches) {
  return dispatchListContainer_.createHandle(std::move(dispatches));
}

} // namespace cut
