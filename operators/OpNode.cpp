#include "OpNode.h"

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

size_t OpNode::executionSize() const {
  auto ds = dispatchSize();
  return static_cast<size_t>(ds.x) * ds.y * ds.z;
}

void OpNode::setHandles(std::vector<Tensor> inputs, Tensor output) {
  inputs_ = std::move(inputs);
  output_ = output;
  hasOutput_ = true;
}

void OpNode::setHandles(std::vector<Tensor> inputs) {
  inputs_ = std::move(inputs);
  hasOutput_ = false;
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

} // namespace cut
