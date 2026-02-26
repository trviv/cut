#pragma once

#include "OpNode.h"

namespace cut {

class UnaryOpNode : public OpNode {
public:
  UnaryOpNode(OperatorEnum op,
              TensorStore &store,
              const Tensor &a,
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

} // namespace cut
