#include "RoPEOp.h"
#include "RoPEShaders.generated.h"
#include "TensorStore.h"

namespace cut {

RoPEOpNode::RoPEOpNode(TensorStore &store,
                       const Tensor &x,
                       const Tensor &cosTable,
                       const Tensor &sinTable,
                       const Tensor &runtimeParams,
                       uint32_t headDim,
                       const Tensor &preallocOutput,
                       std::optional<uint32_t> spec)
    : OpNode(RoPE, store, spec) {
  const auto &buf = store.getTensor(x);
  dtype_ = buf.getDtype();
  outShape_ = buf.getShape();
  numElements_ = static_cast<uint32_t>(actualElementCount(outShape_));
  headDim_ = headDim;
  halfDim_ = headDim / 2;
  inputs_ = {x, cosTable, sinTable, runtimeParams};
  output_ = preallocOutput ? preallocOutput
                           : store.createTensorEmpty(outShape_, dtype_);
}

DataType RoPEOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> RoPEOpNode::shader() const {
  return compiledRoPE(dtype_, dtype_);
}

std::vector<uint32_t> RoPEOpNode::outputShape() const {
  return outShape_;
}

ThreadSize RoPEOpNode::dispatchSize() const {
  uint32_t gridX = ((numElements_ + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> RoPEOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numElements;
    uint32_t headDim;
    uint32_t halfDim;
  } pc{numElements_, headDim_, halfDim_};
  return toBytes(pc);
}

} // namespace cut
