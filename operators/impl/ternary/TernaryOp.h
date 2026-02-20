#pragma once

#include "OpNode.h"

namespace cut {

class TernaryClampOpNode : public OpNode {
public:
  TernaryClampOpNode(std::vector<uint32_t> &&shape,
                     DataType dtype,
                     uint32_t minBits,
                     uint32_t maxBits);

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  uint32_t minBits_;
  uint32_t maxBits_;
  size_t numElements_;
};

class TernarySelectOpNode : public OpNode {
public:
  TernarySelectOpNode(std::vector<uint32_t> &&condShape,
                      std::vector<uint32_t> &&xShape,
                      std::vector<uint32_t> &&yShape,
                      DataType dtype);

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> condShape_;
  std::vector<uint32_t> xShape_;
  std::vector<uint32_t> yShape_;
  DataType dtype_;
  size_t numElements_;
};

} // namespace cut
