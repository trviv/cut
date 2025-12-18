#include <ComputeStructs.h>

namespace cut {

const ThreadGroupSize &ComputeDispatch::threadgroupSize() const {
  return tgSize_;
}

const ComputeHandle &ComputeDispatch::shader() const {
  return shader_;
}

const std::vector<ComputeBinding> &ComputeDispatch::bindings() const {
  return bindings_;
}

void CommandBuffer::encode(ComputeDispatch &&dispatch) {
  encodeImpl(dispatch);
  dispatches_.push_back(std::move(dispatch));
}

} // namespace cut
