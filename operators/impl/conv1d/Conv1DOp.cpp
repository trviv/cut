#include "Conv1DOp.h"
#include "TensorStore.h"

namespace cut {

Conv1DOpNode::Conv1DOpNode(TensorStore &store,
                           const Tensor &input,
                           const Tensor &weight,
                           uint32_t stride,
                           uint32_t padding,
                           std::optional<uint32_t> spec)
    : OpNode(Conv1D, store, spec) {
  const auto &inputBuf = store.getTensor(input);
  const auto &weightBuf = store.getTensor(weight);
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
  spec_ = spec.value_or(kConv1DDefaultVariant);
  inputs_ = {input, weight};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType Conv1DOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> Conv1DOpNode::shader() const {
  return getCompiledConv1D(*spec_, dtype_, dtype_);
}

std::vector<uint32_t> Conv1DOpNode::outputShape() const {
  return {N_, C_out_, L_out_};
}

ThreadSize Conv1DOpNode::dispatchSize() const {
  const auto &info = kConv1DVariants[*spec_];
  if (*spec_ == 0) {
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
