#pragma once

#include "OpNode.h"
#include "impl/conv2d/Conv2DVariants.generated.h"

namespace cut {

class Conv2DOpNode : public OpNode {
public:
  Conv2DOpNode(TensorStore &store,
               const Tensor &input,
               const Tensor &weight,
               uint32_t strideH,
               uint32_t strideW,
               uint32_t padH,
               uint32_t padW,
               std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::vector<DataType>
  resolveInputDtypes(const std::vector<DataType> &inputDtypes) const override;

private:
  uint32_t strideH_, strideW_, padH_, padW_;
  DataType dtype_;
  uint32_t N_, C_in_, H_in_, W_in_, C_out_, kH_, kW_, H_out_, W_out_;
};

} // namespace cut
