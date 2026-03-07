#pragma once

#include "OpNode.h"
#include "impl/matmul/MatMulQ4Variants.generated.h"
#include "impl/matmul/MatMulQ8Variants.generated.h"
#include "impl/matmul/MatMulVariants.generated.h"

namespace cut {

/// Weight quantization format for matmul ops.
enum class QuantFormat { None, Q8, Q4 };

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
  DataType dtypeB_; // B dtype for None, scales dtype for Q8/Q4
  uint32_t M_, K_, N_;
};

} // namespace cut
