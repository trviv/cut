#pragma once

#include "OpNode.h"

namespace cut {

class BinaryVecVecOpNode : public OpNode {
public:
  BinaryVecVecOpNode(OperatorEnum op,
                     std::vector<uint32_t> shapeA,
                     std::vector<uint32_t> shapeB,
                     DataType dtype);

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
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
                        uint32_t scalarBits);

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::vector<ComputeBinding> handleBindings() const override;

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  uint32_t scalarBits_;
  size_t numElements_;
};

} // namespace cut
