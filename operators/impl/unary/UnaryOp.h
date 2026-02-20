#pragma once

#include "OpNode.h"

namespace cut {

class UnaryOpNode : public OpNode {
public:
  UnaryOpNode(OperatorEnum op, std::vector<uint32_t> shape, DataType dtype);

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  size_t numElements_;
};

} // namespace cut
