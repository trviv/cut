#pragma once

#include "OpNode.h"

namespace cut {

class RoPEOpNode : public OpNode {
public:
  RoPEOpNode(TensorStore &store,
             const Tensor &x,
             const Tensor &cosTable,
             const Tensor &sinTable,
             uint32_t pos,
             uint32_t headDim,
             std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t numElements_;
  uint32_t headDim_;
  uint32_t halfDim_;
  uint32_t pos_;
  std::vector<uint32_t> outShape_;
};

} // namespace cut
