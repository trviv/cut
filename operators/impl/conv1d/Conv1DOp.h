#pragma once

#include "OpNode.h"
#include "impl/conv1d/Conv1DVariants.generated.h"

namespace cut {

class Conv1DOpNode : public OpNode {
public:
  Conv1DOpNode(TensorStore &store,
               const Tensor &input,
               const Tensor &weight,
               uint32_t stride,
               uint32_t padding,
               std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  uint32_t stride_, padding_;
  DataType dtype_;
  uint32_t N_, C_in_, L_in_, C_out_, kL_, L_out_;
};

} // namespace cut
