#include "OpNode.h"
#include "Shaders.h"

#include <optional>

namespace cut {

// ============================================================================
// Utility functions
// ============================================================================

size_t alignedElementCount(const std::vector<uint32_t> &shape) {
  if (shape.empty())
    return 0;
  size_t count = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i)
    count *= shape[i];
  size_t alignedInner = (shape.back() + 3) & ~static_cast<uint32_t>(3);
  count *= alignedInner;
  return count;
}

size_t actualElementCount(const std::vector<uint32_t> &shape) {
  if (shape.empty())
    return 0;
  size_t count = 1;
  for (uint32_t dim : shape)
    count *= dim;
  return count;
}

// ============================================================================
// OpNode methods
// ============================================================================

const std::optional<std::vector<uint32_t>> &OpNode::shader() const {
  if (!shader_.has_value()) {
    shader_ = getShader(op_, shaderDtype());
  }
  return shader_;
}

size_t OpNode::shaderKey() const {
  return static_cast<size_t>(op_) | (static_cast<size_t>(shaderDtype()) << 16) |
         (static_cast<size_t>(spec().value_or(0)) << 32);
}

size_t OpNode::executionSize() const {
  auto ds = dispatchSize();
  return static_cast<size_t>(ds.x) * ds.y * ds.z;
}

std::vector<ComputeBinding> OpNode::handleBindings() const {
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

const std::vector<std::unique_ptr<OpNode>> &
OpNode::subOperations(Dispatcher &dispatcher) {
  if (subOps_.empty()) {
    buildSubOperations(dispatcher);
  }
  return subOps_;
}

// ============================================================================
// InternalOpNode
// ============================================================================

InternalOpNode::InternalOpNode(OperatorEnum op,
                               DataType dtype,
                               std::vector<Tensor> inputs,
                               ThreadSize threadSize,
                               std::vector<uint8_t> pushConstants,
                               bool barrierAfter)
    : OpNode(op), dtype_(dtype), threadSize_(threadSize),
      pushConstants_(std::move(pushConstants)), barrierAfter_(barrierAfter) {
  inputs_ = std::move(inputs);
}

DataType InternalOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> InternalOpNode::outputShape() const {
  return {};
}

ThreadSize InternalOpNode::dispatchSize() const {
  return threadSize_;
}

std::vector<uint8_t> InternalOpNode::pushConstants() const {
  return pushConstants_;
}

bool InternalOpNode::needsBarrierAfter() const {
  return barrierAfter_;
}

} // namespace cut
