#include "ScanOp.h"
#include "Runtime.h"

namespace cut {

PrefixScanOpNode::PrefixScanOpNode(OperatorEnum op,
                                   Runtime &runtime,
                                   const Tensor &a,
                                   std::optional<uint32_t> spec)
    : OpNode(op, runtime, spec) {
  const auto &buf = runtime.getTensor(a);
  shape_ = buf.getShape();
  dtype_ = buf.getDtype();
  numElements_ = actualElementCount(shape_);
  inputs_ = {a};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType PrefixScanOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> PrefixScanOpNode::outputShape() const {
  return shape_;
}

bool PrefixScanOpNode::isMultiPass() const {
  return true;
}
size_t PrefixScanOpNode::executionSize() const {
  return numElements_;
}

ThreadSize PrefixScanOpNode::dispatchSize() const {
  return {0, 0, 0};
}
std::vector<uint8_t> PrefixScanOpNode::pushConstants() const {
  return {};
}

} // namespace cut
