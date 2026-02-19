#pragma once

#include "OpNode.h"

namespace cut {

class UnaryOpNode : public OpNode {
public:
  UnaryOpNode(OperatorEnum op, std::vector<uint32_t> shape, DataType dtype)
      : op_(op), shape_(std::move(shape)), dtype_(dtype),
        numElements_(alignedElementCount(shape_)) {}

  void validate() const override {
    // Unary ops have no cross-input validation
  }

  OperatorEnum op() const override { return op_; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return shape_; }

  ThreadSize dispatchSize() const override {
    return {static_cast<uint32_t>(numElements_), 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    uint32_t n = static_cast<uint32_t>(numElements_);
    return toBytes(n);
  }

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  size_t numElements_;
};

} // namespace cut
