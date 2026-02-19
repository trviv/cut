#include "SortOp.h"

namespace cut {

// --- BitonicSortOpNode ---

BitonicSortOpNode::BitonicSortOpNode(size_t executionSize, DataType dtype)
    : executionSize_(executionSize), dtype_(dtype) {}

void BitonicSortOpNode::validate() const {}

OperatorEnum BitonicSortOpNode::op() const {
  return SortBitonic;
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

RadixSortOpNode::RadixSortOpNode(size_t executionSize, DataType dtype)
    : executionSize_(executionSize), dtype_(dtype) {}

void RadixSortOpNode::validate() const {}

OperatorEnum RadixSortOpNode::op() const {
  return SortRadix;
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
