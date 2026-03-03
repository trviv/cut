#include "MaxPool2DOp.h"
#include "TensorStore.h"

namespace cut {

MaxPool2DOpNode::MaxPool2DOpNode(TensorStore &store,
                                 const Tensor &input,
                                 uint32_t kernelH,
                                 uint32_t kernelW,
                                 uint32_t strideH,
                                 uint32_t strideW,
                                 uint32_t padH,
                                 uint32_t padW,
                                 std::optional<uint32_t> spec)
    : OpNode(MaxPool2D, store, spec) {
  const auto &buf = store.getTensor(input);
  dtype_ = buf.getDtype();
  auto shape = buf.getShape();
  if (shape.size() != 4)
    throw std::runtime_error("maxPool2d: input must be 4D [N, C, H, W]");
  kernelH_ = kernelH;
  kernelW_ = kernelW;
  strideH_ = strideH;
  strideW_ = strideW;
  padH_ = padH;
  padW_ = padW;
  N_ = shape[0];
  C_ = shape[1];
  H_in_ = shape[2];
  W_in_ = shape[3];
  H_out_ = (H_in_ + 2 * padH_ - kernelH_) / strideH_ + 1;
  W_out_ = (W_in_ + 2 * padW_ - kernelW_) / strideW_ + 1;
  spec_ = spec.value_or(kMaxPool2DDefaultVariant);
  inputs_ = {input};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType MaxPool2DOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> MaxPool2DOpNode::shader() const {
  return getCompiledMaxPool2D(*spec_, dtype_, dtype_);
}

std::vector<uint32_t> MaxPool2DOpNode::outputShape() const {
  return {N_, C_, H_out_, W_out_};
}

ThreadSize MaxPool2DOpNode::dispatchSize() const {
  const auto &info = kMaxPool2DVariants[*spec_];
  if (*spec_ == 0) {
    // Naive variant: original dispatch logic (vec4 output)
    uint32_t outAlignedW4 = ((W_out_ + 3) & ~3u) / 4;
    uint32_t gridX = ((outAlignedW4 + info.wgX - 1) / info.wgX) * info.wgX;
    uint32_t gridY = ((N_ * C_ * H_out_ + info.wgY - 1) / info.wgY) * info.wgY;
    return {gridX, gridY, 1};
  }
  // Tiled: x = tiles along W_out, y = tiles along H_out, z = N * C
  uint32_t gridX = ((W_out_ + info.effTileN - 1) / info.effTileN) * info.wgX;
  uint32_t gridY = ((H_out_ + info.effTileM - 1) / info.effTileM) * info.wgY;
  uint32_t gridZ = N_ * C_;
  return {gridX, gridY, gridZ};
}

std::vector<uint8_t> MaxPool2DOpNode::pushConstants() const {
  struct PoolParams {
    uint32_t N, C, H_in, W_in;
    uint32_t kernelH, kernelW;
    uint32_t strideH, strideW;
    uint32_t padH, padW;
  } pc{N_,       C_,       H_in_,    W_in_, kernelH_,
       kernelW_, strideH_, strideW_, padH_, padW_};
  return toBytes(pc);
}

} // namespace cut
