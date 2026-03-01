#include "CreationOp.h"
#include "CreationShaders.generated.h"
#include "Shaders.h"
#include "TensorStore.h"

namespace cut {

// --- FillOpNode ---

FillOpNode::FillOpNode(OperatorEnum op,
                       TensorStore &store,
                       std::vector<uint32_t> &&shape,
                       DataType dtype,
                       float fillValue,
                       std::optional<uint32_t> spec)
    : OpNode(op, store, spec), shape_(std::move(shape)), dtype_(dtype),
      fillValue_(fillValue), numElements_(alignedElementCount(shape_)) {
  if (op_ == Ones)
    fillValue_ = 1.0f;
}

DataType FillOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> FillOpNode::shader() const {
  auto compiled = compiledFill(dtype_, dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
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

// --- ArangeOpNode ---

ArangeOpNode::ArangeOpNode(OperatorEnum op,
                           TensorStore &store,
                           std::vector<uint32_t> &&shape,
                           DataType dtype,
                           float start,
                           float step,
                           std::optional<uint32_t> spec)
    : OpNode(op, store, spec), shape_(std::move(shape)), dtype_(dtype),
      start_(start), step_(step), numElements_(alignedElementCount(shape_)) {}

DataType ArangeOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> ArangeOpNode::shader() const {
  auto compiled = compiledArange(dtype_, dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
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

} // namespace cut
