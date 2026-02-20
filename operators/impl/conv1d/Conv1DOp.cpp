#include "Conv1DOp.h"
#include "Runtime.h"

namespace cut {

Conv1DOpNode::Conv1DOpNode(Runtime &runtime,
                           const Tensor &input,
                           const Tensor &weight,
                           uint32_t stride,
                           uint32_t padding,
                           std::optional<uint32_t> spec)
    : OpNode(Conv1D, runtime, spec) {
  const auto &inputBuf = runtime.getTensor(input);
  const auto &weightBuf = runtime.getTensor(weight);
  const auto inShape = inputBuf.getShape();
  const auto wShape = weightBuf.getShape();
  stride_ = stride;
  padding_ = padding;
  dtype_ = inputBuf.getDtype();
  if (inShape.size() != 3)
    throw std::runtime_error("conv1d: input must be 3D [N, C_in, L_in]");
  if (wShape.size() != 3)
    throw std::runtime_error("conv1d: weight must be 3D [C_out, C_in, kL]");
  N_ = inShape[0];
  C_in_ = inShape[1];
  L_in_ = inShape[2];
  C_out_ = wShape[0];
  kL_ = wShape[2];
  if (wShape[1] != C_in_)
    throw std::runtime_error("conv1d: weight C_in dimension mismatch");
  L_out_ = (L_in_ + 2 * padding_ - kL_) / stride_ + 1;
  resolvedVariant_ = spec.value_or(kConv1DDefaultVariant);
  inputs_ = {input, weight};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType Conv1DOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<uint32_t> Conv1DOpNode::spec() const {
  return resolvedVariant_;
}

std::vector<uint32_t> Conv1DOpNode::outputShape() const {
  return {N_, C_out_, L_out_};
}

ThreadSize Conv1DOpNode::dispatchSize() const {
  const auto &info = kConv1DVariants[resolvedVariant_];
  if (resolvedVariant_ == 0) {
    // Naive: linear dispatch over all output elements
    uint32_t totalOutputs = N_ * C_out_ * L_out_;
    uint32_t gridX = ((totalOutputs + info.wgX - 1) / info.wgX) * info.wgX;
    return {gridX, 1, 1};
  }
  // Tiled: x = tiles along L_out, y = N * C_out
  uint32_t gridX = ((L_out_ + info.effTileN - 1) / info.effTileN) * info.wgX;
  uint32_t gridY = N_ * C_out_;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> Conv1DOpNode::pushConstants() const {
  struct Conv1DParams {
    uint32_t batchSize, C_in, L_in, C_out, kL;
    uint32_t stride, padding;
  } pc{N_, C_in_, L_in_, C_out_, kL_, stride_, padding_};
  return toBytes(pc);
}

} // namespace cut
