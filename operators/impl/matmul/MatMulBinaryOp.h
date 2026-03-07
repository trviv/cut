#pragma once

#include "OpNode.h"
#include "impl/matmul/MatMulOp.h"

namespace cut {

/// Fused dequantize-matmul with binary vector operation.
///
/// Computes binaryOp(A * dequant(B), D).
/// Supports Q8 and Q4 quantization formats (auto-detected from shapes).
///
/// Inputs:
///   0: A          — Float32 activations [M, K]
///   1: packedB    — Int8 quantized/packed values
///   2: scalesB    — Float16 per-block scales [ceil(K/32), N]
///   3: D          — Float32 binary operand [M, N]
/// Output:
///   Float32 [M, N]
class MatMulBinaryOpNode : public OpNode {
public:
  MatMulBinaryOpNode(TensorStore &store,
                     OperatorEnum binaryOp,
                     const Tensor &a,
                     const Tensor &packedB,
                     const Tensor &scalesB,
                     const Tensor &d,
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
  OperatorEnum binaryOp_;
  DataType dtypeA_;
  DataType dtypeScales_;
  uint32_t M_, K_, N_;
};

} // namespace cut
