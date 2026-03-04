#pragma once

#include "OpNode.h"

namespace cut {

/// Fused RMS normalization.
///
/// Computes: output = x * rsqrt(mean(x²) + eps) * weight
///
/// Inputs:
///   0: x      — Float32/Float16 vector [dim] (input to normalize)
///   1: weight — Float32/Float16 vector [dim] (normalization weight)
/// Output:
///   Float32/Float16 [dim] (same as input dtype)
///
/// This fuses 4 operations into 1 GPU dispatch:
///   1. UnarySquare(x) → squared
///   2. ReduceSum(squared) → sum_squares
///   3. VecScalarMul(rsqrt(sum/dim + eps)) → scale
///   4. VecVecMul(x * scale, weight) → output
class RMSNormOpNode : public OpNode {
public:
  RMSNormOpNode(TensorStore &store,
                const Tensor &x,
                const Tensor &weight,
                float eps = 1e-5f,
                std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::string displayName() const override;

private:
  DataType dtype_;
  uint32_t dim_;
  uint32_t alignedDim_;
  float eps_;
};

} // namespace cut
