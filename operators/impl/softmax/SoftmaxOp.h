#pragma once

#include "OpNode.h"

namespace cut {

class SoftmaxOpNode : public OpNode {
public:
  /// op should be Softmax or LogSoftmax.
  SoftmaxOpNode(OperatorEnum op,
                TensorStore &store,
                const Tensor &a,
                int dim,
                std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  size_t shaderKey() const override;

private:
  DataType dtype_;
  int dim_;
  uint32_t outerSize_;
  uint32_t reduceSize_;
  uint32_t innerSize_;
  uint32_t inOuterStride_;
  uint32_t inReduceStride_;
  uint32_t bufInnerDim_;
  uint32_t alignedBufInner_;
  std::vector<uint32_t> outShape_;
};

} // namespace cut
