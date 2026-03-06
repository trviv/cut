#pragma once

#include "OpNode.h"

namespace cut {

/// Fused dequantize-matmul for Q4_0 quantized weights.
///
/// Computes C = A * B with inline dequantization:
///   C[m][n] = sum_k( A[m][k] * (nibble_B[k][n] - 8) * scale[k/32][n] )
///
/// Inputs:
///   0: A          — Float32 activations [M, K]
///   1: packedB    — Int8 packed nibbles [K, N/2] (two 4-bit values per byte)
///   2: scalesB    — Float16 per-block scales [ceil(K/32), N]
/// Output:
///   Float32 [M, N]
class MatMulQ4OpNode : public OpNode {
public:
  MatMulQ4OpNode(TensorStore &store,
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
  DataType dtypeA_;
  DataType dtypeScales_;
  uint32_t M_, K_, N_;
};

} // namespace cut
