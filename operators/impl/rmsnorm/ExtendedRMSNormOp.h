#pragma once

#include "OpNode.h"

namespace cut {

/// Fused residual addition + RMS normalization.
///
/// Computes: output = (residual_base + delta) * rsqrt(mean((residual_base +
/// delta)²) + eps) * weight
///
/// Inputs:
///   0: residual_base — Float32/Float16 vector [dim] (e.g., previous hidden
///   state) 1: delta          — Float32/Float16 vector [dim] (e.g.,
///   attention/FFN output) 2: weight         — Float32/Float16 vector [dim]
///   (normalization weight)
/// Output:
///   Float32/Float16 [dim] (same as input dtype)
///
/// This fuses 5 operations into 1 GPU dispatch:
///   1. VecVecAdd(residual_base, delta) → residual
///   2. UnarySquare(residual) → squared
///   3. ReduceSum(squared) → sum_squares
///   4. VecScalarMul(rsqrt(sum/dim + eps)) → scale
///   5. VecVecMul(residual * scale, weight) → output
class ExtendedRMSNormOpNode : public OpNode {
public:
  ExtendedRMSNormOpNode(TensorStore &store,
                        const Tensor &residual_base,
                        const Tensor &delta,
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
