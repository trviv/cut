#include <ComputeStructs.h>

namespace cut {

void CommandBuffer::encode(ComputeDispatch &&dispatch) {
  encodeImpl(dispatch);
  dispatches_.push_back(std::move(dispatch));
}

} // namespace cut
