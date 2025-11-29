#include <ComputeInterface.h>

namespace cut {

ComputeHandle ComputeInterface::registerDispatch(ComputeDispatch &&dispatch) {
  if (dispatch.referenceDispatchHandle_ &&
      dispatchContainer_.get(dispatch.referenceDispatchHandle_)
          .referenceDispatchHandle_) {
    logErr("Creating dispatch to a reference which itself is a reference, "
           "is not allowed.");
  }

  return dispatchContainer_.createHandle(std::move(dispatch));
}

} // namespace cut
