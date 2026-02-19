#pragma once

#include "OpNode.h"

namespace cut {

class PrefixScanOpNode : public OpNode {
public:
  PrefixScanOpNode(OperatorEnum op, std::vector<uint32_t> shape, DataType dtype)
      : op_(op), shape_(std::move(shape)), dtype_(dtype),
        numElements_(actualElementCount(shape_)) {}

  void validate() const override {}

  OperatorEnum op() const override { return op_; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return shape_; }

  bool isMultiPass() const override { return true; }
  size_t executionSize() const override { return numElements_; }

  // Multi-pass ops have per-pass dispatch; these are not used directly
  ThreadSize dispatchSize() const override { return {0, 0, 0}; }
  std::vector<uint8_t> pushConstants() const override { return {}; }

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  size_t numElements_;
};

} // namespace cut
