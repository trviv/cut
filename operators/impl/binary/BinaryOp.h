#pragma once

#include "OpNode.h"

namespace cut {

class BinaryVecVecOpNode : public OpNode {
public:
  BinaryVecVecOpNode(OperatorEnum op,
                     std::vector<uint32_t> shapeA,
                     std::vector<uint32_t> shapeB,
                     DataType dtype)
      : op_(op), shapeA_(std::move(shapeA)), shapeB_(std::move(shapeB)),
        dtype_(dtype), numElements_(alignedElementCount(shapeA_)) {}

  void validate() const override {
    if (actualElementCount(shapeA_) != actualElementCount(shapeB_)) {
      throw std::runtime_error(
          "Size mismatch: " + std::to_string(actualElementCount(shapeA_)) +
          " vs " + std::to_string(actualElementCount(shapeB_)));
    }
  }

  OperatorEnum op() const override { return op_; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return shapeA_; }

  ThreadSize dispatchSize() const override {
    return {static_cast<uint32_t>(numElements_), 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    uint32_t n = static_cast<uint32_t>(numElements_);
    return toBytes(n);
  }

private:
  OperatorEnum op_;
  std::vector<uint32_t> shapeA_;
  std::vector<uint32_t> shapeB_;
  DataType dtype_;
  size_t numElements_;
};

class BinaryVecScalarOpNode : public OpNode {
public:
  BinaryVecScalarOpNode(OperatorEnum op,
                        std::vector<uint32_t> shape,
                        DataType dtype,
                        uint32_t scalarBits)
      : op_(op), shape_(std::move(shape)), dtype_(dtype),
        scalarBits_(scalarBits), numElements_(alignedElementCount(shape_)) {}

  void validate() const override {
    // Vec-scalar ops have no cross-input validation
  }

  OperatorEnum op() const override { return op_; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return shape_; }

  ThreadSize dispatchSize() const override {
    return {static_cast<uint32_t>(numElements_), 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    struct PushConstants {
      uint32_t numElements;
      uint32_t scalarBits;
    } pc{static_cast<uint32_t>(numElements_), scalarBits_};
    return toBytes(pc);
  }

  std::vector<ComputeBinding> handleBindings() const override {
    // Vec-scalar: input at 0, output at 1 (scalar is in push constants)
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

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  uint32_t scalarBits_;
  size_t numElements_;
};

} // namespace cut
