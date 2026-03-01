#include "UnaryOp.h"
#include "Shaders.h"
#include "TensorStore.h"
#include "UnaryShaders.generated.h"

namespace cut {

UnaryOpNode::UnaryOpNode(OperatorEnum op,
                         TensorStore &store,
                         const Tensor &a,
                         std::optional<uint32_t> spec)
    : OpNode(op, store, spec) {
  const auto &buf = store.getTensor(a);
  dtype_ = buf.getDtype();
  numElements_ = alignedElementCount(buf.getShape());
  inputs_ = {a};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType UnaryOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> UnaryOpNode::shader() const {
  auto compiled = compiledUnary(dtype_, dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> UnaryOpNode::outputShape() const {
  return store_->getTensor(inputs_[0]).getShape();
}

ThreadSize UnaryOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> UnaryOpNode::pushConstants() const {
  uint32_t n = static_cast<uint32_t>(numElements_);
  return toBytes(n);
}

} // namespace cut
