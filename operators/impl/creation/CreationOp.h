#pragma once

#include "OpNode.h"

namespace cut {

class FillOpNode : public OpNode {
public:
  FillOpNode(OperatorEnum op,
             std::vector<uint32_t> shape,
             DataType dtype,
             float fillValue = 0.0f)
      : op_(op), shape_(std::move(shape)), dtype_(dtype), fillValue_(fillValue),
        numElements_(alignedElementCount(shape_)) {
    if (op_ == Ones)
      fillValue_ = 1.0f;
  }

  void validate() const override {}

  OperatorEnum op() const override { return op_; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return shape_; }

  ThreadSize dispatchSize() const override {
    return {static_cast<uint32_t>(numElements_), 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    struct PushConstants {
      uint32_t numElements;
      float fillValue;
    } pc{static_cast<uint32_t>(numElements_), fillValue_};
    return toBytes(pc);
  }

  // Fill ops have only an output, no input
  std::vector<ComputeBinding> handleBindings() const override {
    std::vector<ComputeBinding> bindings;
    if (hasOutput_) {
      bindings.emplace_back(0u, output_);
    }
    return bindings;
  }

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  float fillValue_;
  size_t numElements_;
};

class ArangeOpNode : public OpNode {
public:
  ArangeOpNode(OperatorEnum op,
               std::vector<uint32_t> shape,
               DataType dtype,
               float start,
               float step)
      : op_(op), shape_(std::move(shape)), dtype_(dtype), start_(start),
        step_(step), numElements_(alignedElementCount(shape_)) {}

  void validate() const override {}

  OperatorEnum op() const override { return op_; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return shape_; }

  ThreadSize dispatchSize() const override {
    return {static_cast<uint32_t>(numElements_), 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    struct PushConstants {
      uint32_t numElements;
      float start;
      float step;
    } pc{static_cast<uint32_t>(numElements_), start_, step_};
    return toBytes(pc);
  }

  // Arange/Linspace have only an output, no input
  std::vector<ComputeBinding> handleBindings() const override {
    std::vector<ComputeBinding> bindings;
    if (hasOutput_) {
      bindings.emplace_back(0u, output_);
    }
    return bindings;
  }

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  float start_;
  float step_;
  size_t numElements_;
};

} // namespace cut
