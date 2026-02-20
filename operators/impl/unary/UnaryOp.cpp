#include "UnaryOp.h"
#include "Runtime.h"

namespace cut {

UnaryOpNode::UnaryOpNode(OperatorEnum op, Runtime &runtime, const Tensor &a)
    : OpNode(op, runtime) {
  const auto &buf = runtime.getTensor(a);
  shape_ = buf.getShape();
  dtype_ = buf.getDtype();
  numElements_ = alignedElementCount(shape_);
  inputs_ = {a};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType UnaryOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> UnaryOpNode::outputShape() const {
  return shape_;
}

ThreadSize UnaryOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> UnaryOpNode::pushConstants() const {
  uint32_t n = static_cast<uint32_t>(numElements_);
  return toBytes(n);
}

} // namespace cut
