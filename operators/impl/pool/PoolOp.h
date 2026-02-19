#pragma once

#include "OpNode.h"

namespace cut {

class MaxPool2DOpNode : public OpNode {
public:
  MaxPool2DOpNode(std::vector<uint32_t> inShape,
                  uint32_t kernelH,
                  uint32_t kernelW,
                  uint32_t strideH,
                  uint32_t strideW,
                  uint32_t padH,
                  uint32_t padW,
                  DataType dtype)
      : inShape_(std::move(inShape)), kernelH_(kernelH), kernelW_(kernelW),
        strideH_(strideH), strideW_(strideW), padH_(padH), padW_(padW),
        dtype_(dtype) {
    N_ = inShape_[0];
    C_ = inShape_[1];
    H_in_ = inShape_[2];
    W_in_ = inShape_[3];
    H_out_ = (H_in_ + 2 * padH_ - kernelH_) / strideH_ + 1;
    W_out_ = (W_in_ + 2 * padW_ - kernelW_) / strideW_ + 1;
  }

  void validate() const override {
    if (inShape_.size() != 4)
      throw std::runtime_error("max_pool2d: input must be 4D [N, C, H, W]");
  }

  OperatorEnum op() const override { return MaxPool2D; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override {
    return {N_, C_, H_out_, W_out_};
  }

  ThreadSize dispatchSize() const override {
    const uint32_t tileSize = 16;
    uint32_t outAlignedW4 = ((W_out_ + 3) & ~3u) / 4;
    uint32_t gridX = (outAlignedW4 + tileSize - 1) / tileSize * tileSize;
    uint32_t gridY = (N_ * C_ * H_out_ + tileSize - 1) / tileSize * tileSize;
    return {gridX, gridY, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    struct Pool2DParams {
      uint32_t N, C, H_in, W_in;
      uint32_t kernelH, kernelW;
      uint32_t strideH, strideW;
      uint32_t padH, padW;
    } pc{N_,       C_,       H_in_,    W_in_, kernelH_,
         kernelW_, strideH_, strideW_, padH_, padW_};
    return toBytes(pc);
  }

private:
  std::vector<uint32_t> inShape_;
  uint32_t kernelH_, kernelW_, strideH_, strideW_, padH_, padW_;
  DataType dtype_;
  uint32_t N_, C_, H_in_, W_in_, H_out_, W_out_;
};

class AvgPool2DOpNode : public OpNode {
public:
  AvgPool2DOpNode(std::vector<uint32_t> inShape,
                  uint32_t kernelH,
                  uint32_t kernelW,
                  uint32_t strideH,
                  uint32_t strideW,
                  uint32_t padH,
                  uint32_t padW,
                  DataType dtype)
      : inShape_(std::move(inShape)), kernelH_(kernelH), kernelW_(kernelW),
        strideH_(strideH), strideW_(strideW), padH_(padH), padW_(padW),
        dtype_(dtype) {
    N_ = inShape_[0];
    C_ = inShape_[1];
    H_in_ = inShape_[2];
    W_in_ = inShape_[3];
    H_out_ = (H_in_ + 2 * padH_ - kernelH_) / strideH_ + 1;
    W_out_ = (W_in_ + 2 * padW_ - kernelW_) / strideW_ + 1;
  }

  void validate() const override {
    if (inShape_.size() != 4)
      throw std::runtime_error("avg_pool2d: input must be 4D [N, C, H, W]");
  }

  OperatorEnum op() const override { return AvgPool2D; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override {
    return {N_, C_, H_out_, W_out_};
  }

  ThreadSize dispatchSize() const override {
    const uint32_t tileSize = 16;
    uint32_t outAlignedW4 = ((W_out_ + 3) & ~3u) / 4;
    uint32_t gridX = (outAlignedW4 + tileSize - 1) / tileSize * tileSize;
    uint32_t gridY = (N_ * C_ * H_out_ + tileSize - 1) / tileSize * tileSize;
    return {gridX, gridY, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    struct Pool2DParams {
      uint32_t N, C, H_in, W_in;
      uint32_t kernelH, kernelW;
      uint32_t strideH, strideW;
      uint32_t padH, padW;
    } pc{N_,       C_,       H_in_,    W_in_, kernelH_,
         kernelW_, strideH_, strideW_, padH_, padW_};
    return toBytes(pc);
  }

private:
  std::vector<uint32_t> inShape_;
  uint32_t kernelH_, kernelW_, strideH_, strideW_, padH_, padW_;
  DataType dtype_;
  uint32_t N_, C_, H_in_, W_in_, H_out_, W_out_;
};

} // namespace cut
