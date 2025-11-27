#include <ComputeContainers.h>

namespace cut {

ComputeDispatch::ComputeDispatch(
    const ComputeHandle &shader, const ThreadGroupSize &tgSize,
    const ComputeHandle &referenceDispatchHandle,
    const std::vector<ComputeHandle> &resourceBindings,
    const std::vector<DataReference> &dataReferences)
    : shader_(shader), tgSize_(tgSize),
      referenceDispatchHandle_(referenceDispatchHandle),
      resourceBindings_(resourceBindings) {
  dataBindings_.resize(dataReferences.size());

  for (uint32_t i = 0; i < dataReferences.size(); i++) {
    const auto *bytePtr = static_cast<const uint8_t *>(dataReferences[i].ptr);
    dataBindings_[i] = {bytePtr, bytePtr + dataReferences[i].size};
  }
}

void ComputeDispatch::bindShader(const ComputeHandle &shaderHandle) {
  shader_ = shaderHandle;
}

void ComputeDispatch::bindResource(const ComputeHandle &resourceHandle,
                                   uint32_t index) {
  if (resourceBindings_.size() <= index) {
    resourceBindings_.resize(index + 1, {});
  }

  resourceBindings_[index] = resourceHandle;
}

void ComputeDispatch::bindData(const DataReference &data, uint32_t index) {
  if (data.ptr == nullptr) {
    return;
  }

  if (dataBindings_.size() <= index) {
    dataBindings_.resize(index + 1, {});
  }

  const auto bytePtr = static_cast<const uint8_t *>(data.ptr);
  dataBindings_[index].assign(bytePtr, bytePtr + data.size);
}

void ComputeDispatch::setThreadGroupSize(const ThreadGroupSize &tgSize) {
  tgSize_ = tgSize;
}

} // namespace cut
