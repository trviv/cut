#pragma once

#include "OpNode.h"

namespace cut {

/// Fused matrix multiplication with SiLU activation: silu(A * B)
/// where silu(x) = x / (1 + exp(-x))
///
/// Saves GPU dispatch overhead vs separate matmul + silu operations.
/// Used in FFN gate projections (30× per forward pass in SmolLM2).
class MatMulSiLUOpNode : public OpNode {
public:
  MatMulSiLUOpNode(TensorStore &store,
                   const Tensor &a,
                   const Tensor &b,
                   std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  size_t shaderKey() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::vector<DataType>
  resolveInputDtypes(const std::vector<DataType> &inputDtypes) const override;

private:
  DataType dtypeA_;
  DataType dtypeB_;
  uint32_t M_, K_, N_;
};

} // namespace cut
