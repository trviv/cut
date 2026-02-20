#include "ScanOp.h"

namespace cut {

PrefixScanOpNode::PrefixScanOpNode(OperatorEnum op,
                                   std::vector<uint32_t> shape,
                                   DataType dtype)
    : op_(op), shape_(std::move(shape)), dtype_(dtype),
      numElements_(actualElementCount(shape_)) {}

OperatorEnum PrefixScanOpNode::op() const {
  return op_;
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
