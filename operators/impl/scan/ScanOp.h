#pragma once

#include "OpNode.h"

namespace cut {

class PrefixScanOpNode : public OpNode {
public:
  PrefixScanOpNode(OperatorEnum op,
                   std::vector<uint32_t> shape,
                   DataType dtype);

  OperatorEnum op() const override;
  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  bool isMultiPass() const override;
  size_t executionSize() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  size_t numElements_;
};

} // namespace cut
