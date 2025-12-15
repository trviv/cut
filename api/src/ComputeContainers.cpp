#include <ComputeContainers.h>

namespace cut {

ComputeDispatch::ComputeDispatch(const ComputeHandle &shader,
                                 const ThreadGroupSize &tgSize,
                                 const std::vector<ComputeBinding> &bindings)
    : shader_(shader), tgSize_(tgSize), bindings_(bindings) {}

void ComputeDispatch::bindShader(const ComputeHandle &shaderHandle) {
  shader_ = shaderHandle;
}

void ComputeDispatch::bindResource(const ComputeHandle &resourceHandle,
                                   uint32_t index) {
  bindings_.emplace_back(static_cast<int32_t>(index), resourceHandle);
}

void ComputeDispatch::bindData(const DataReference &data, uint32_t index) {
  if (data.ptr == nullptr) {
    return;
  }

  bindings_.emplace_back(static_cast<int32_t>(index), data);
}

void ComputeDispatch::setThreadGroupSize(const ThreadGroupSize &tgSize) {
  tgSize_ = tgSize;
}

} // namespace cut
