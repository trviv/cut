#include "Conv2DOp.h"
#include "Runtime.h"

namespace cut {

Conv2DOpNode::Conv2DOpNode(Runtime &runtime,
                           const Tensor &input,
                           const Tensor &weight,
                           uint32_t strideH,
                           uint32_t strideW,
                           uint32_t padH,
                           uint32_t padW,
                           std::optional<uint32_t> spec)
    : OpNode(Conv2D, runtime, spec) {
  const auto &inputBuf = runtime.getTensor(input);
  const auto &weightBuf = runtime.getTensor(weight);
  const auto inShape = inputBuf.getShape();
  const auto wShape = weightBuf.getShape();
  strideH_ = strideH;
  strideW_ = strideW;
  padH_ = padH;
  padW_ = padW;
  dtype_ = inputBuf.getDtype();
  if (inShape.size() != 4)
    throw std::runtime_error("conv2d: input must be 4D [N, C_in, H_in, W_in]");
  if (wShape.size() != 4)
    throw std::runtime_error("conv2d: weight must be 4D [C_out, C_in, kH, kW]");
  N_ = inShape[0];
  C_in_ = inShape[1];
  H_in_ = inShape[2];
  W_in_ = inShape[3];
  C_out_ = wShape[0];
  kH_ = wShape[2];
  kW_ = wShape[3];
  if (wShape[1] != C_in_)
    throw std::runtime_error("conv2d: weight C_in dimension mismatch");
  H_out_ = (H_in_ + 2 * padH_ - kH_) / strideH_ + 1;
  W_out_ = (W_in_ + 2 * padW_ - kW_) / strideW_ + 1;
  spec_ = spec.value_or(kConv2DDefaultVariant);
  inputs_ = {input, weight};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
}

DataType Conv2DOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> Conv2DOpNode::shader() const {
  return getCompiledConv2D(*spec_, dtype_);
}

std::vector<uint32_t> Conv2DOpNode::outputShape() const {
  return {N_, C_out_, H_out_, W_out_};
}

ThreadSize Conv2DOpNode::dispatchSize() const {
  const auto &info = kConv2DVariants[*spec_];
  if (*spec_ == 0) {
    // Naive: x = W_out, y = linearized(N*C_out*H_out)
    uint32_t gridX = ((W_out_ + info.effTileN - 1) / info.effTileN) * info.wgX;
    uint32_t gridY =
        ((N_ * C_out_ * H_out_ + info.effTileM - 1) / info.effTileM) * info.wgY;
    return {gridX, gridY, 1};
  }
  // Tiled: x = tiles along W_out, y = tiles along H_out, z = N * C_out
  uint32_t gridX = ((W_out_ + info.effTileN - 1) / info.effTileN) * info.wgX;
  uint32_t gridY = ((H_out_ + info.effTileM - 1) / info.effTileM) * info.wgY;
  uint32_t gridZ = N_ * C_out_;
  return {gridX, gridY, gridZ};
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
