#pragma once

#include "OpNode.h"

namespace cut {

class Conv1DOpNode : public OpNode {
public:
  Conv1DOpNode(std::vector<uint32_t> inShape,
               std::vector<uint32_t> wShape,
               uint32_t stride,
               uint32_t padding,
               DataType dtype);

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> inShape_;
  std::vector<uint32_t> wShape_;
  uint32_t stride_, padding_;
  DataType dtype_;
  uint32_t N_, C_in_, L_in_, C_out_, kL_, L_out_;
};

class Conv2DOpNode : public OpNode {
public:
  Conv2DOpNode(std::vector<uint32_t> inShape,
               std::vector<uint32_t> wShape,
               uint32_t strideH,
               uint32_t strideW,
               uint32_t padH,
               uint32_t padW,
               DataType dtype);

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> inShape_;
  std::vector<uint32_t> wShape_;
  uint32_t strideH_, strideW_, padH_, padW_;
  DataType dtype_;
  uint32_t N_, C_in_, H_in_, W_in_, C_out_, kH_, kW_, H_out_, W_out_;
};

} // namespace cut
