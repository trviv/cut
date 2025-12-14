#include <ComputeStructs.h>

namespace cut {

void CommandBuffer::encode(const ComputeHandle &dispatchHandle) {
  dispatches_.push_back(dispatchHandle);
  encodeImpl(dispatchHandle);
}

} // namespace cut
