#pragma once

#include "OpNode.h"

namespace cut {

class BinaryVecVecOpNode : public OpNode {
public:
  BinaryVecVecOpNode(OperatorEnum op,
                     Runtime &runtime,
                     const Tensor &a,
                     const Tensor &b,
                     std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  size_t numElements_;
};

class BinaryVecScalarOpNode : public OpNode {
public:
  BinaryVecScalarOpNode(OperatorEnum op,
                        Runtime &runtime,
                        const Tensor &a,
                        const ComputeData &b,
                        std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::optional<std::vector<uint32_t>> shader() const override;

private:
  DataType dtype_;
  uint32_t scalarBits_{};
  size_t numElements_;
  bool usesHandleScalar_ = false;
};

} // namespace cut
