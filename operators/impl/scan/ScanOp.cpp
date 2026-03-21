#include "ScanOp.h"
#include "TensorStore.h"

namespace cut {

PrefixScanOpNode::PrefixScanOpNode(OperatorEnum op,
                                   TensorStore &store,
                                   const Tensor &a,
                                   std::optional<uint32_t> spec)
    : OpNode(op, store, spec) {
  const auto &buf = store.getTensor(a);
  dtype_ = buf.getDtype();
  numElements_ = actualElementCount(buf.getShape());
  inputs_ = {a};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType PrefixScanOpNode::outputDtype() const {
  return dtype_;
}

std::vector<uint32_t> PrefixScanOpNode::outputShape() const {
  return store_->getTensor(inputs_[0]).getShape();
}

bool PrefixScanOpNode::isMultiPass() const {
  return true;
}
size_t PrefixScanOpNode::executionSize() const {
  return numElements_;
}

ThreadSize PrefixScanOpNode::dispatchSize() const {
  return {0, 0, 0};
}
std::vector<uint8_t> PrefixScanOpNode::pushConstants() const {
  return {};
}

void PrefixScanOpNode::buildSubOperations() {
  uint32_t numElements = static_cast<uint32_t>(numElements_);
  uint32_t isExclusive = (op_ == PrefixScanExclusiveSum) ? 1u : 0u;
  uint32_t groupCount = (numElements + 255) / 256;

  Tensor inputHandle = inputs_[0];
  Tensor outputHandle = output_;

  struct ScanPC {
    uint32_t numElements;
    uint32_t isExclusive;
  } scanPC{numElements, isExclusive};

  if (groupCount <= 1) {
    // Single workgroup: simple scan
    Tensor partialSums = store_->acquireTempBuffer(1, DataType::Float32);
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalScanPerWg, DataType::Float32,
        std::vector<Tensor>{inputHandle, outputHandle, partialSums},
        outputHandle, ThreadSize{256, 1, 1}, toBytes(scanPC)));
    return;
  }

  // Multi-workgroup: three-pass approach
  Tensor partialSums = store_->acquireTempBuffer(groupCount, DataType::Float32);

  // Pass 1: Per-workgroup scan
  subOps_.push_back(std::make_unique<InternalOpNode>(
      InternalScanPerWg, DataType::Float32,
      std::vector<Tensor>{inputHandle, outputHandle, partialSums}, outputHandle,
      ThreadSize{256 * groupCount, 1, 1}, toBytes(scanPC), true));

  // Pass 2: Exclusive scan on partial sums (single thread)
  subOps_.push_back(std::make_unique<InternalOpNode>(
      InternalScanPartialSums, DataType::Float32,
      std::vector<Tensor>{partialSums}, partialSums, ThreadSize{1, 1, 1},
      toBytes(groupCount), true));

  // Pass 3: Add group prefix to each element
  subOps_.push_back(std::make_unique<InternalOpNode>(
      InternalScanPropagate, DataType::Float32,
      std::vector<Tensor>{partialSums, outputHandle}, outputHandle,
      ThreadSize{256 * groupCount, 1, 1}, toBytes(numElements)));
}

} // namespace cut
