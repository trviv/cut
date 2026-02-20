#include "BinaryOp.h"
#include "Runtime.h"

namespace cut {

// --- BinaryVecVecOpNode ---

BinaryVecVecOpNode::BinaryVecVecOpNode(OperatorEnum op,
                                       Runtime &runtime,
                                       const Tensor &a,
                                       const Tensor &b)
    : OpNode(op, runtime) {
  const auto &bufA = runtime.getTensor(a);
  const auto &bufB = runtime.getTensor(b);
  shapeA_ = bufA.getShape();
  shapeB_ = bufB.getShape();
  dtype_ = bufA.getDtype();
  numElements_ = alignedElementCount(shapeA_);
  if (actualElementCount(shapeA_) != actualElementCount(shapeB_)) {
    throw std::runtime_error(
        "Size mismatch: " + std::to_string(actualElementCount(shapeA_)) +
        " vs " + std::to_string(actualElementCount(shapeB_)));
  }
  inputs_ = {a, b};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
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
                                             Runtime &runtime,
                                             const Tensor &a,
                                             uint32_t scalarBits)
    : OpNode(op, runtime) {
  const auto &buf = runtime.getTensor(a);
  shape_ = buf.getShape();
  dtype_ = buf.getDtype();
  scalarBits_ = scalarBits;
  numElements_ = alignedElementCount(shape_);
  inputs_ = {a};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
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
