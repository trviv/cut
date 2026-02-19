#include "UnaryOp.h"

namespace cut {

UnaryOpNode::UnaryOpNode(OperatorEnum op,
                         std::vector<uint32_t> shape,
                         DataType dtype)
    : op_(op), shape_(std::move(shape)), dtype_(dtype),
      numElements_(alignedElementCount(shape_)) {}

void UnaryOpNode::validate() const {
  // Unary ops have no cross-input validation
}

OperatorEnum UnaryOpNode::op() const {
  return op_;
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
