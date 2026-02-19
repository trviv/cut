#include "TernaryOp.h"

namespace cut {

// --- TernaryClampOpNode ---

TernaryClampOpNode::TernaryClampOpNode(std::vector<uint32_t> shape,
                                       DataType dtype,
                                       uint32_t minBits,
                                       uint32_t maxBits)
    : shape_(std::move(shape)), dtype_(dtype), minBits_(minBits),
      maxBits_(maxBits), numElements_(alignedElementCount(shape_)) {}

void TernaryClampOpNode::validate() const {}

OperatorEnum TernaryClampOpNode::op() const {
  return TernaryClamp;
}
DataType TernaryClampOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> TernaryClampOpNode::outputShape() const {
  return shape_;
}

ThreadSize TernaryClampOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> TernaryClampOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numElements;
    uint32_t minBits;
    uint32_t maxBits;
  } pc{static_cast<uint32_t>(numElements_), minBits_, maxBits_};
  return toBytes(pc);
}

// --- TernarySelectOpNode ---

TernarySelectOpNode::TernarySelectOpNode(std::vector<uint32_t> condShape,
                                         std::vector<uint32_t> xShape,
                                         std::vector<uint32_t> yShape,
                                         DataType dtype)
    : condShape_(std::move(condShape)), xShape_(std::move(xShape)),
      yShape_(std::move(yShape)), dtype_(dtype),
      numElements_(alignedElementCount(xShape_)) {}

void TernarySelectOpNode::validate() const {
  if (actualElementCount(condShape_) != actualElementCount(xShape_) ||
      actualElementCount(condShape_) != actualElementCount(yShape_)) {
    throw std::runtime_error("condition, x, and y must have the same size");
  }
}

OperatorEnum TernarySelectOpNode::op() const {
  return TernarySelect;
}
DataType TernarySelectOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> TernarySelectOpNode::outputShape() const {
  return xShape_;
}

ThreadSize TernarySelectOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> TernarySelectOpNode::pushConstants() const {
  uint32_t n = static_cast<uint32_t>(numElements_);
  return toBytes(n);
}

} // namespace cut
