#include "BinaryOp.h"

namespace cut {

// --- BinaryVecVecOpNode ---

BinaryVecVecOpNode::BinaryVecVecOpNode(OperatorEnum op,
                                       std::vector<uint32_t> shapeA,
                                       std::vector<uint32_t> shapeB,
                                       DataType dtype)
    : op_(op), shapeA_(std::move(shapeA)), shapeB_(std::move(shapeB)),
      dtype_(dtype), numElements_(alignedElementCount(shapeA_)) {
  if (actualElementCount(shapeA_) != actualElementCount(shapeB_)) {
    throw std::runtime_error(
        "Size mismatch: " + std::to_string(actualElementCount(shapeA_)) +
        " vs " + std::to_string(actualElementCount(shapeB_)));
  }
}

OperatorEnum BinaryVecVecOpNode::op() const {
  return op_;
}
DataType BinaryVecVecOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> BinaryVecVecOpNode::outputShape() const {
  return shapeA_;
}

ThreadSize BinaryVecVecOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> BinaryVecVecOpNode::pushConstants() const {
  uint32_t n = static_cast<uint32_t>(numElements_);
  return toBytes(n);
}

// --- BinaryVecScalarOpNode ---

BinaryVecScalarOpNode::BinaryVecScalarOpNode(OperatorEnum op,
                                             std::vector<uint32_t> shape,
                                             DataType dtype,
                                             uint32_t scalarBits)
    : op_(op), shape_(std::move(shape)), dtype_(dtype), scalarBits_(scalarBits),
      numElements_(alignedElementCount(shape_)) {}

OperatorEnum BinaryVecScalarOpNode::op() const {
  return op_;
}
DataType BinaryVecScalarOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> BinaryVecScalarOpNode::outputShape() const {
  return shape_;
}

ThreadSize BinaryVecScalarOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> BinaryVecScalarOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numElements;
    uint32_t scalarBits;
  } pc{static_cast<uint32_t>(numElements_), scalarBits_};
  return toBytes(pc);
}

std::vector<ComputeBinding> BinaryVecScalarOpNode::handleBindings() const {
  // Vec-scalar: input at 0, output at 1 (scalar is in push constants)
  std::vector<ComputeBinding> bindings;
  uint32_t idx = 0;
  for (const auto &h : inputs_) {
    bindings.emplace_back(idx++, h);
  }
  if (hasOutput_) {
    bindings.emplace_back(idx++, output_);
  }
  return bindings;
}

} // namespace cut
