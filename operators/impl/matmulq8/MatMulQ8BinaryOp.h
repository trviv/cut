#pragma once

#include "OpNode.h"

namespace cut {

/// Fused dequantize-matmul with binary vector operation for Q8_0 quantized
/// weights.
///
/// Computes binaryOp(A * dequant(B), D):
///   C[m][n] = binaryOp( sum_k( A[m][k] * int8_B[k][n] * scale[k/32][n] ),
///                        D[m][n] )
///
/// Inputs:
///   0: A          — Float32 activations [M, K]
///   1: packedB    — Int8 quantized values [K, N] (transposed at load time)
///   2: scalesB    — Float16 per-block scales [ceil(K/32), N]
///   3: D          — Float32 binary operand [M, N]
/// Output:
///   Float32 [M, N]
///
/// The binary operation is selected by the binaryOp enum (e.g. BinaryAdd,
/// BinaryMul) and applied via a specialization constant in the shader.
class MatMulQ8BinaryOpNode : public OpNode {
public:
  MatMulQ8BinaryOpNode(TensorStore &store,
                       OperatorEnum binaryOp,
                       const Tensor &a,
                       const Tensor &packedB,
                       const Tensor &scalesB,
                       const Tensor &d,
                       uint32_t bCols,
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
  OperatorEnum binaryOp_;
  DataType dtypeA_;
  DataType dtypeScales_;
  uint32_t M_, K_, N_;
};

} // namespace cut
