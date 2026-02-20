#include "BinaryOp.h"
#include "Runtime.h"

namespace cut {

// --- BinaryVecVecOpNode ---

BinaryVecVecOpNode::BinaryVecVecOpNode(OperatorEnum op,
                                       Runtime &runtime,
                                       const Tensor &a,
                                       const Tensor &b,
                                       std::optional<uint32_t> spec)
    : OpNode(op, runtime, spec) {
  const auto &bufA = runtime.getTensor(a);
  const auto &bufB = runtime.getTensor(b);
  const auto shapeA = bufA.getShape();
  const auto shapeB = bufB.getShape();
  dtype_ = bufA.getDtype();
  numElements_ = alignedElementCount(shapeA);
  if (actualElementCount(shapeA) != actualElementCount(shapeB)) {
    throw std::runtime_error(
        "Size mismatch: " + std::to_string(actualElementCount(shapeA)) +
        " vs " + std::to_string(actualElementCount(shapeB)));
  }
  inputs_ = {a, b};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType BinaryVecVecOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> BinaryVecVecOpNode::outputShape() const {
  return runtime_->getTensor(inputs_[0]).getShape();
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
                                             uint32_t scalarBits,
                                             std::optional<uint32_t> spec)
    : OpNode(op, runtime, spec) {
  const auto &buf = runtime.getTensor(a);
  dtype_ = buf.getDtype();
  scalarBits_ = scalarBits;
  numElements_ = alignedElementCount(buf.getShape());
  inputs_ = {a};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType BinaryVecScalarOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> BinaryVecScalarOpNode::outputShape() const {
  return runtime_->getTensor(inputs_[0]).getShape();
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
