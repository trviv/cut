#pragma once

#include "OpNode.h"
#include "impl/matmul/MatMulVariants.generated.h"

namespace cut {

class MatMulOpNode : public OpNode {
public:
  MatMulOpNode(TensorStore &store,
               const Tensor &a,
               const Tensor &b,
               std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtypeA_;
  DataType dtypeB_;
  uint32_t M_, K_, N_;
};

} // namespace cut
