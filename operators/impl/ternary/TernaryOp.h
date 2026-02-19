#pragma once

#include "OpNode.h"

namespace cut {

class TernaryClampOpNode : public OpNode {
public:
  TernaryClampOpNode(std::vector<uint32_t> shape,
                     DataType dtype,
                     uint32_t minBits,
                     uint32_t maxBits)
      : shape_(std::move(shape)), dtype_(dtype), minBits_(minBits),
        maxBits_(maxBits), numElements_(alignedElementCount(shape_)) {}

  void validate() const override {}

  OperatorEnum op() const override { return TernaryClamp; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return shape_; }

  ThreadSize dispatchSize() const override {
    return {static_cast<uint32_t>(numElements_), 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    struct PushConstants {
      uint32_t numElements;
      uint32_t minBits;
      uint32_t maxBits;
    } pc{static_cast<uint32_t>(numElements_), minBits_, maxBits_};
    return toBytes(pc);
  }

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  uint32_t minBits_;
  uint32_t maxBits_;
  size_t numElements_;
};

class TernarySelectOpNode : public OpNode {
public:
  TernarySelectOpNode(std::vector<uint32_t> condShape,
                      std::vector<uint32_t> xShape,
                      std::vector<uint32_t> yShape,
                      DataType dtype)
      : condShape_(std::move(condShape)), xShape_(std::move(xShape)),
        yShape_(std::move(yShape)), dtype_(dtype),
        numElements_(alignedElementCount(xShape_)) {}

  void validate() const override {
    if (actualElementCount(condShape_) != actualElementCount(xShape_) ||
        actualElementCount(condShape_) != actualElementCount(yShape_)) {
      throw std::runtime_error("condition, x, and y must have the same size");
    }
  }

  OperatorEnum op() const override { return TernarySelect; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return xShape_; }

  ThreadSize dispatchSize() const override {
    return {static_cast<uint32_t>(numElements_), 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    uint32_t n = static_cast<uint32_t>(numElements_);
    return toBytes(n);
  }

private:
  std::vector<uint32_t> condShape_;
  std::vector<uint32_t> xShape_;
  std::vector<uint32_t> yShape_;
  DataType dtype_;
  size_t numElements_;
};

} // namespace cut
