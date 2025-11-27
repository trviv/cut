#include <ComputeInterface.h>

namespace cut {

ComputeHandle
ComputeInterface::createDispatch(const ComputeHandle &shader,
                                 const ThreadGroupSize &tgSize,
                                 const std::vector<ComputeBinding> &bindings,
                                 const ComputeHandle &refDispatchHandle) {
  if (refDispatchHandle &&
      dispatchContainer_.get(refDispatchHandle)->referenceDispatchHandle_) {
    logErr("Creating dispatch to a reference which itself is a reference, "
           "is not allowed.");
  }

  return dispatchContainer_.createHandle(
      {shader, tgSize, bindings, refDispatchHandle});
}

ComputeHandle ComputeInterface::createDispatchList(DispatchList &&dispatches) {
  return dispatchListContainer_.createHandle(std::move(dispatches));
}

} // namespace cut
