#include "TernaryOp.h"
#include "Runtime.h"

namespace cut {

// --- TernaryClampOpNode ---

TernaryClampOpNode::TernaryClampOpNode(Runtime &runtime,
                                       const Tensor &a,
                                       uint32_t minBits,
                                       uint32_t maxBits,
                                       std::optional<uint32_t> spec)
    : OpNode(TernaryClamp, runtime, spec) {
  const auto &buf = runtime.getTensor(a);
  shape_ = buf.getShape();
  dtype_ = buf.getDtype();
  minBits_ = minBits;
  maxBits_ = maxBits;
  numElements_ = alignedElementCount(shape_);
  inputs_ = {a};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
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

TernarySelectOpNode::TernarySelectOpNode(Runtime &runtime,
                                         const Tensor &cond,
                                         const Tensor &x,
                                         const Tensor &y,
                                         std::optional<uint32_t> spec)
    : OpNode(TernarySelect, runtime, spec) {
  const auto &condBuf = runtime.getTensor(cond);
  const auto &xBuf = runtime.getTensor(x);
  const auto &yBuf = runtime.getTensor(y);
  condShape_ = condBuf.getShape();
  xShape_ = xBuf.getShape();
  yShape_ = yBuf.getShape();
  dtype_ = xBuf.getDtype();
  numElements_ = alignedElementCount(xShape_);
  if (actualElementCount(condShape_) != actualElementCount(xShape_) ||
      actualElementCount(condShape_) != actualElementCount(yShape_)) {
    throw std::runtime_error("condition, x, and y must have the same size");
  }
  inputs_ = {cond, x, y};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
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
