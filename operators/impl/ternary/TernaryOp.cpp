#include "TernaryOp.h"
#include "Runtime.h"
#include "Shaders.h"
#include "TernaryShaders.generated.h"

namespace cut {

// --- TernaryClampOpNode ---

TernaryClampOpNode::TernaryClampOpNode(Runtime &runtime,
                                       const Tensor &a,
                                       uint32_t minBits,
                                       uint32_t maxBits,
                                       std::optional<uint32_t> spec)
    : OpNode(TernaryClamp, runtime, spec) {
  const auto &buf = runtime.getTensor(a);
  dtype_ = buf.getDtype();
  minBits_ = minBits;
  maxBits_ = maxBits;
  numElements_ = alignedElementCount(buf.getShape());
  inputs_ = {a};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
}

DataType TernaryClampOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> TernaryClampOpNode::shader() const {
  auto compiled = compiledTernaryClamp(dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> TernaryClampOpNode::outputShape() const {
  return runtime_->getTensor(inputs_[0]).getShape();
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
  const auto condShape = condBuf.getShape();
  const auto xShape = xBuf.getShape();
  const auto yShape = yBuf.getShape();
  dtype_ = xBuf.getDtype();
  numElements_ = alignedElementCount(xShape);
  if (actualElementCount(condShape) != actualElementCount(xShape) ||
      actualElementCount(condShape) != actualElementCount(yShape)) {
    throw std::runtime_error("condition, x, and y must have the same size");
  }
  inputs_ = {cond, x, y};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
}

DataType TernarySelectOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> TernarySelectOpNode::shader() const {
  auto compiled = compiledTernarySelect(dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> TernarySelectOpNode::outputShape() const {
  return runtime_->getTensor(inputs_[1]).getShape();
}

ThreadSize TernarySelectOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> TernarySelectOpNode::pushConstants() const {
  uint32_t n = static_cast<uint32_t>(numElements_);
  return toBytes(n);
}

} // namespace cut
