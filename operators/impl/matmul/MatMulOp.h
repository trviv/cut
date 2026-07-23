#pragma once

#include "OpNode.h"
#include "impl/matmul/MatMulQ4Variants.generated.h"
#include "impl/matmul/MatMulQ8Variants.generated.h"
#include "impl/matmul/MatMulVariants.generated.h"

namespace cut {

/// Weight quantization format for matmul ops.
enum class QuantFormat { None, Q8, Q4 };

/// Fusion mode for matmul ops (applied at output write via SPIR-V linking).
enum class MatMulFusion { None, Unary, Binary };

class MatMulOpNode : public OpNode {
public:
  /// Standard matmul: C = A * B (2 inputs)
  MatMulOpNode(TensorStore &store,
               const Tensor &a,
               const Tensor &b,
               std::optional<uint32_t> spec = {});

  /// Quantized matmul: C = A * dequant(packedB, scalesB) (3 inputs)
  /// Auto-detects Q4 vs Q8 from input shapes.
  MatMulOpNode(TensorStore &store,
               const Tensor &a,
               const Tensor &packedB,
               const Tensor &scalesB,
               std::optional<uint32_t> spec = {});

  QuantFormat format() const { return format_; }
  MatMulFusion fusion() const { return fusion_; }

  /// Set fusion mode after construction (called by optimizer passes).
  /// fusionOp is the unary or binary op code to apply.
  /// For Binary fusion, d is the [M,N] tensor to combine with the matmul
  /// result.
  void setFusion(MatMulFusion fusion,
                 OperatorEnum fusionOp = {},
                 const Tensor &d = {});

  DataType outputDtype() const override;
  size_t shaderKey() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::vector<ComputeBinding> bindings() const override;
  std::string displayName() const override;
  std::vector<DataType>
  resolveInputDtypes(const std::vector<DataType> &inputDtypes) const override;

private:
  QuantFormat format_;
  MatMulFusion fusion_ = MatMulFusion::None;
  OperatorEnum fusionOp_{};
  DataType dtypeA_;
  DataType dtypeB_; // B dtype for None, scales dtype for Q8/Q4
  uint32_t M_, K_, N_;
  // True when the variant was auto-selected (no explicit spec from the caller).
  bool autoSpec_ = false;
  /// True when the Q4 scales tensor also carries per-block mins in its lower
  /// half (affine / zero-point 4-bit). Drives specialization constant 3.
  bool q4Affine_ = false;
  // Fusion-capable variant to fall back to if fusion is enabled while a
  // cooperative-matrix variant is selected (CoopMat shaders have no fusion
  // epilogue). Computed at construction alongside the primary selection.
  uint32_t fusionFallbackSpec_ = 0;
};

} // namespace cut
