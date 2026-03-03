#pragma once

#include "OpNode.h"

namespace cut {

/// Fused dequantize-matmul for Q8_0 quantized weights.
///
/// Computes C = A * B^T with inline dequantization:
///   C[m][n] = sum_k( A[m][k] * int8_B[n][k] * scale[n][k/32] )
///
/// Inputs:
///   0: A          — Float32 activations [M, K]
///   1: packedB    — Int8 quantized values [N, K] (non-transposed GGUF layout)
///   2: scalesB    — Float16 per-block scales [N, ceil(K/32)]
/// Output:
///   Float32 [M, N]
class MatMulQ8OpNode : public OpNode {
public:
  MatMulQ8OpNode(TensorStore &store,
                 const Tensor &a,
                 const Tensor &packedB,
                 const Tensor &scalesB,
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
  DataType dtypeA_;
  uint32_t M_, K_, N_;
};

} // namespace cut
