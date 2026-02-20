#include "SortOp.h"
#include "Runtime.h"

namespace cut {

// --- BitonicSortOpNode ---

BitonicSortOpNode::BitonicSortOpNode(Runtime &runtime,
                                     const Tensor &keys,
                                     const Tensor &vals)
    : OpNode(SortBitonic, runtime) {
  const auto &buf = runtime.getTensor(keys);
  executionSize_ = actualElementCount(buf.getShape());
  dtype_ = buf.getDtype();
  inputs_ = {keys, vals};
}

DataType BitonicSortOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> BitonicSortOpNode::outputShape() const {
  return {};
}

bool BitonicSortOpNode::isMultiPass() const {
  return true;
}
size_t BitonicSortOpNode::executionSize() const {
  return executionSize_;
}

ThreadSize BitonicSortOpNode::dispatchSize() const {
  return {0, 0, 0};
}
std::vector<uint8_t> BitonicSortOpNode::pushConstants() const {
  return {};
}

// --- RadixSortOpNode ---

RadixSortOpNode::RadixSortOpNode(Runtime &runtime,
                                 const Tensor &keys,
                                 const Tensor &vals)
    : OpNode(SortRadix, runtime) {
  const auto &buf = runtime.getTensor(keys);
  executionSize_ = actualElementCount(buf.getShape());
  dtype_ = buf.getDtype();
  inputs_ = {keys, vals};
}

DataType RadixSortOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> RadixSortOpNode::outputShape() const {
  return {};
}

bool RadixSortOpNode::isMultiPass() const {
  return true;
}
size_t RadixSortOpNode::executionSize() const {
  return executionSize_;
}

ThreadSize RadixSortOpNode::dispatchSize() const {
  return {0, 0, 0};
}
std::vector<uint8_t> RadixSortOpNode::pushConstants() const {
  return {};
}

} // namespace cut
