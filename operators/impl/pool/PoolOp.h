#pragma once

#include "OpNode.h"

namespace cut {

class MaxPool2DOpNode : public OpNode {
public:
  MaxPool2DOpNode(Runtime &runtime,
                  const Tensor &input,
                  uint32_t kernelH,
                  uint32_t kernelW,
                  uint32_t strideH,
                  uint32_t strideW,
                  uint32_t padH,
                  uint32_t padW,
                  std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  uint32_t kernelH_, kernelW_, strideH_, strideW_, padH_, padW_;
  DataType dtype_;
  uint32_t N_, C_, H_in_, W_in_, H_out_, W_out_;
};

class AvgPool2DOpNode : public OpNode {
public:
  AvgPool2DOpNode(Runtime &runtime,
                  const Tensor &input,
                  uint32_t kernelH,
                  uint32_t kernelW,
                  uint32_t strideH,
                  uint32_t strideW,
                  uint32_t padH,
                  uint32_t padW,
                  std::optional<uint32_t> spec = {});

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  uint32_t kernelH_, kernelW_, strideH_, strideW_, padH_, padW_;
  DataType dtype_;
  uint32_t N_, C_, H_in_, W_in_, H_out_, W_out_;
};

} // namespace cut
