#pragma once

#include "OpNode.h"

namespace cut {

class TernaryClampOpNode : public OpNode {
public:
  TernaryClampOpNode(Runtime &runtime,
                     const Tensor &a,
                     uint32_t minBits,
                     uint32_t maxBits,
                     std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t minBits_;
  uint32_t maxBits_;
  size_t numElements_;
};

class TernarySelectOpNode : public OpNode {
public:
  TernarySelectOpNode(Runtime &runtime,
                      const Tensor &cond,
                      const Tensor &x,
                      const Tensor &y,
                      std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  size_t numElements_;
};

} // namespace cut
