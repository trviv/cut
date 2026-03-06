#pragma once

#include "OpNode.h"

namespace cut {

/// Quantization format for GPU dequantization.
enum class DequantFormat : uint32_t {
  BF16 = 0,
  Q4_K = 1,
  Q5_K = 2,
  Q6_K = 3,
};

/// GPU dequantization op: converts raw quantized bytes to Float32.
/// Input: 1D Int8 tensor of raw GGUF block data.
/// Output: 2D Float32 tensor [rows, cols].
class DequantOpNode : public OpNode {
public:
  DequantOpNode(TensorStore &store,
                const Tensor &rawData,
                DequantFormat format,
                uint32_t rows,
                uint32_t cols,
                std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  size_t shaderKey() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::string displayName() const override;

private:
  DequantFormat format_;
  uint32_t rows_;
  uint32_t cols_;
  uint32_t outputStride_; // aligned cols (multiple of 4)
};

} // namespace cut
