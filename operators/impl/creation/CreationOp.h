#pragma once

#include "OpNode.h"

namespace cut {

class FillOpNode : public OpNode {
public:
  FillOpNode(OperatorEnum op,
             Runtime &runtime,
             std::vector<uint32_t> &&shape,
             DataType dtype,
             float fillValue = 0.0f,
             std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  float fillValue_;
  size_t numElements_;
};

class ArangeOpNode : public OpNode {
public:
  ArangeOpNode(OperatorEnum op,
               Runtime &runtime,
               std::vector<uint32_t> &&shape,
               DataType dtype,
               float start,
               float step,
               std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  float start_;
  float step_;
  size_t numElements_;
};

} // namespace cut
