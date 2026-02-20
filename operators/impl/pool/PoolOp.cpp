#include "PoolOp.h"
#include "Runtime.h"

namespace cut {

// --- MaxPool2DOpNode ---

MaxPool2DOpNode::MaxPool2DOpNode(Runtime &runtime,
                                 const Tensor &input,
                                 uint32_t kernelH,
                                 uint32_t kernelW,
                                 uint32_t strideH,
                                 uint32_t strideW,
                                 uint32_t padH,
                                 uint32_t padW,
                                 std::optional<uint32_t> spec)
    : OpNode(MaxPool2D, runtime, spec) {
  const auto &buf = runtime.getTensor(input);
  const auto inShape = buf.getShape();
  dtype_ = buf.getDtype();
  kernelH_ = kernelH;
  kernelW_ = kernelW;
  strideH_ = strideH;
  strideW_ = strideW;
  padH_ = padH;
  padW_ = padW;
  if (inShape.size() != 4)
    throw std::runtime_error("max_pool2d: input must be 4D [N, C, H, W]");
  N_ = inShape[0];
  C_ = inShape[1];
  H_in_ = inShape[2];
  W_in_ = inShape[3];
  H_out_ = (H_in_ + 2 * padH_ - kernelH_) / strideH_ + 1;
  W_out_ = (W_in_ + 2 * padW_ - kernelW_) / strideW_ + 1;
  inputs_ = {input};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType MaxPool2DOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> MaxPool2DOpNode::outputShape() const {
  return {N_, C_, H_out_, W_out_};
}

ThreadSize MaxPool2DOpNode::dispatchSize() const {
  const uint32_t tileSize = 16;
  uint32_t outAlignedW4 = ((W_out_ + 3) & ~3u) / 4;
  uint32_t gridX = (outAlignedW4 + tileSize - 1) / tileSize * tileSize;
  uint32_t gridY = (N_ * C_ * H_out_ + tileSize - 1) / tileSize * tileSize;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> MaxPool2DOpNode::pushConstants() const {
  struct Pool2DParams {
    uint32_t N, C, H_in, W_in;
    uint32_t kernelH, kernelW;
    uint32_t strideH, strideW;
    uint32_t padH, padW;
  } pc{N_,       C_,       H_in_,    W_in_, kernelH_,
       kernelW_, strideH_, strideW_, padH_, padW_};
  return toBytes(pc);
}

// --- AvgPool2DOpNode ---

AvgPool2DOpNode::AvgPool2DOpNode(Runtime &runtime,
                                 const Tensor &input,
                                 uint32_t kernelH,
                                 uint32_t kernelW,
                                 uint32_t strideH,
                                 uint32_t strideW,
                                 uint32_t padH,
                                 uint32_t padW,
                                 std::optional<uint32_t> spec)
    : OpNode(AvgPool2D, runtime, spec) {
  const auto &buf = runtime.getTensor(input);
  const auto inShape = buf.getShape();
  dtype_ = buf.getDtype();
  kernelH_ = kernelH;
  kernelW_ = kernelW;
  strideH_ = strideH;
  strideW_ = strideW;
  padH_ = padH;
  padW_ = padW;
  if (inShape.size() != 4)
    throw std::runtime_error("avg_pool2d: input must be 4D [N, C, H, W]");
  N_ = inShape[0];
  C_ = inShape[1];
  H_in_ = inShape[2];
  W_in_ = inShape[3];
  H_out_ = (H_in_ + 2 * padH_ - kernelH_) / strideH_ + 1;
  W_out_ = (W_in_ + 2 * padW_ - kernelW_) / strideW_ + 1;
  inputs_ = {input};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType AvgPool2DOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> AvgPool2DOpNode::outputShape() const {
  return {N_, C_, H_out_, W_out_};
}

ThreadSize AvgPool2DOpNode::dispatchSize() const {
  const uint32_t tileSize = 16;
  uint32_t outAlignedW4 = ((W_out_ + 3) & ~3u) / 4;
  uint32_t gridX = (outAlignedW4 + tileSize - 1) / tileSize * tileSize;
  uint32_t gridY = (N_ * C_ * H_out_ + tileSize - 1) / tileSize * tileSize;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> AvgPool2DOpNode::pushConstants() const {
  struct Pool2DParams {
    uint32_t N, C, H_in, W_in;
    uint32_t kernelH, kernelW;
    uint32_t strideH, strideW;
    uint32_t padH, padW;
  } pc{N_,       C_,       H_in_,    W_in_, kernelH_,
       kernelW_, strideH_, strideW_, padH_, padW_};
  return toBytes(pc);
}

} // namespace cut
