#include "CreationOp.h"
#include "Runtime.h"

namespace cut {

// --- FillOpNode ---

FillOpNode::FillOpNode(OperatorEnum op,
                       Runtime &runtime,
                       std::vector<uint32_t> &&shape,
                       DataType dtype,
                       float fillValue)
    : OpNode(op, runtime), shape_(std::move(shape)), dtype_(dtype),
      fillValue_(fillValue), numElements_(alignedElementCount(shape_)) {
  if (op_ == Ones)
    fillValue_ = 1.0f;
}

DataType FillOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> FillOpNode::outputShape() const {
  return shape_;
}

ThreadSize FillOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> FillOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numElements;
    float fillValue;
  } pc{static_cast<uint32_t>(numElements_), fillValue_};
  return toBytes(pc);
}

std::vector<ComputeBinding> FillOpNode::handleBindings() const {
  std::vector<ComputeBinding> bindings;
  if (hasOutput_) {
    bindings.emplace_back(0u, output_);
  }
  return bindings;
}

// --- ArangeOpNode ---

ArangeOpNode::ArangeOpNode(OperatorEnum op,
                           Runtime &runtime,
                           std::vector<uint32_t> &&shape,
                           DataType dtype,
                           float start,
                           float step)
    : OpNode(op, runtime), shape_(std::move(shape)), dtype_(dtype),
      start_(start), step_(step), numElements_(alignedElementCount(shape_)) {}

DataType ArangeOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> ArangeOpNode::outputShape() const {
  return shape_;
}

ThreadSize ArangeOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> ArangeOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numElements;
    float start;
    float step;
  } pc{static_cast<uint32_t>(numElements_), start_, step_};
  return toBytes(pc);
}

std::vector<ComputeBinding> ArangeOpNode::handleBindings() const {
  std::vector<ComputeBinding> bindings;
  if (hasOutput_) {
    bindings.emplace_back(0u, output_);
  }
  return bindings;
}

} // namespace cut
