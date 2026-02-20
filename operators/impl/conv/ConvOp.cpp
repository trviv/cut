#include "ConvOp.h"

namespace cut {

// --- Conv1DOpNode ---

Conv1DOpNode::Conv1DOpNode(std::vector<uint32_t> inShape,
                           std::vector<uint32_t> wShape,
                           uint32_t stride,
                           uint32_t padding,
                           DataType dtype)
    : inShape_(std::move(inShape)), wShape_(std::move(wShape)), stride_(stride),
      padding_(padding), dtype_(dtype) {
  if (inShape_.size() != 3)
    throw std::runtime_error("conv1d: input must be 3D [N, C_in, L_in]");
  if (wShape_.size() != 3)
    throw std::runtime_error("conv1d: weight must be 3D [C_out, C_in, kL]");
  N_ = inShape_[0];
  C_in_ = inShape_[1];
  L_in_ = inShape_[2];
  C_out_ = wShape_[0];
  kL_ = wShape_[2];
  if (wShape_[1] != C_in_)
    throw std::runtime_error("conv1d: weight C_in dimension mismatch");
  L_out_ = (L_in_ + 2 * padding_ - kL_) / stride_ + 1;
}

OperatorEnum Conv1DOpNode::op() const {
  return Conv1D;
}
DataType Conv1DOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> Conv1DOpNode::outputShape() const {
  return {N_, C_out_, L_out_};
}

ThreadSize Conv1DOpNode::dispatchSize() const {
  uint32_t totalOutputs = N_ * C_out_ * L_out_;
  uint32_t gridX = ((totalOutputs + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> Conv1DOpNode::pushConstants() const {
  struct Conv1DParams {
    uint32_t batchSize, C_in, L_in, C_out, kL;
    uint32_t stride, padding;
  } pc{N_, C_in_, L_in_, C_out_, kL_, stride_, padding_};
  return toBytes(pc);
}

// --- Conv2DOpNode ---

Conv2DOpNode::Conv2DOpNode(std::vector<uint32_t> inShape,
                           std::vector<uint32_t> wShape,
                           uint32_t strideH,
                           uint32_t strideW,
                           uint32_t padH,
                           uint32_t padW,
                           DataType dtype)
    : inShape_(std::move(inShape)), wShape_(std::move(wShape)),
      strideH_(strideH), strideW_(strideW), padH_(padH), padW_(padW),
      dtype_(dtype) {
  if (inShape_.size() != 4)
    throw std::runtime_error("conv2d: input must be 4D [N, C_in, H_in, W_in]");
  if (wShape_.size() != 4)
    throw std::runtime_error("conv2d: weight must be 4D [C_out, C_in, kH, kW]");
  N_ = inShape_[0];
  C_in_ = inShape_[1];
  H_in_ = inShape_[2];
  W_in_ = inShape_[3];
  C_out_ = wShape_[0];
  kH_ = wShape_[2];
  kW_ = wShape_[3];
  if (wShape_[1] != C_in_)
    throw std::runtime_error("conv2d: weight C_in dimension mismatch");
  H_out_ = (H_in_ + 2 * padH_ - kH_) / strideH_ + 1;
  W_out_ = (W_in_ + 2 * padW_ - kW_) / strideW_ + 1;
}

OperatorEnum Conv2DOpNode::op() const {
  return Conv2D;
}
DataType Conv2DOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> Conv2DOpNode::outputShape() const {
  return {N_, C_out_, H_out_, W_out_};
}

ThreadSize Conv2DOpNode::dispatchSize() const {
  const uint32_t tileSize = 16;
  uint32_t gridX = (W_out_ + tileSize - 1) / tileSize * tileSize;
  uint32_t gridY = (N_ * C_out_ * H_out_ + tileSize - 1) / tileSize * tileSize;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> Conv2DOpNode::pushConstants() const {
  struct Conv2DParams {
    uint32_t batchSize, C_in, H_in, W_in;
    uint32_t C_out, kH, kW;
    uint32_t strideH, strideW;
    uint32_t padH, padW;
  } pc{N_,  C_in_,    H_in_,    W_in_, C_out_, kH_,
       kW_, strideH_, strideW_, padH_, padW_};
  return toBytes(pc);
}

} // namespace cut
