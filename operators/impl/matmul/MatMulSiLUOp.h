#pragma once

#include "OpNode.h"
#include "impl/matmul/MatMulOp.h"

namespace cut {

/// Fused matrix multiplication with SiLU activation: silu(A * B)
/// where silu(x) = x / (1 + exp(-x))
///
/// Supports all weight formats (None, Q8, Q4).
class MatMulSiLUOpNode : public OpNode {
public:
  /// Standard matmul + SiLU (2 inputs)
  MatMulSiLUOpNode(TensorStore &store,
                   const Tensor &a,
                   const Tensor &b,
                   std::optional<uint32_t> spec = {});

  /// Quantized matmul + SiLU (3 inputs, auto-detect Q4/Q8)
  MatMulSiLUOpNode(TensorStore &store,
                   const Tensor &a,
                   const Tensor &packedB,
                   const Tensor &scalesB,
                   std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  size_t shaderKey() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::string displayName() const override;
  std::vector<DataType>
  resolveInputDtypes(const std::vector<DataType> &inputDtypes) const override;

private:
  QuantFormat format_;
  DataType dtypeA_;
  DataType dtypeB_;
  uint32_t M_, K_, N_;
};

} // namespace cut
