#include "Conv2DOp.h"
#include "TensorStore.h"

namespace cut {

Conv2DOpNode::Conv2DOpNode(TensorStore &store,
                           const Tensor &input,
                           const Tensor &weight,
                           uint32_t strideH,
                           uint32_t strideW,
                           uint32_t padH,
                           uint32_t padW,
                           std::optional<uint32_t> spec)
    : OpNode(Conv2D, store, spec) {
  const auto &inputBuf = store.getTensor(input);
  const auto &weightBuf = store.getTensor(weight);
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
  spec_ = spec.value_or(defaultVariant(store));
  // A pinned variant that cannot express this window would read past its shared
  // tile, so fall back the way the scan op does rather than dispatch it.
  if (!conv2dVariantSupportsShape(static_cast<int>(*spec_), kH_, kW_, strideH_,
                                  strideW_))
    spec_ = defaultVariant(store);
  inputs_ = {input, weight};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

uint32_t Conv2DOpNode::defaultVariant(TensorStore &store) const {
  // Vulkan stays on the direct variant: the implicit-GEMM HLSL is the same
  // algorithm but single-buffered and scalar, and has not been measured there.
  // On CUDA the GEMM path is the default at every shape — the direct kernels
  // re-read each input pixel and each weight once per output element, which
  // costs an order of magnitude even on the shapes that suit them best.
  if (store.caps().backend != ComputeBackend::CUDA)
    return kConv2DDefaultVariant;

  int wide = -1, tall = -1, narrow = -1, spatial = -1;
  for (int i = 0; i < kConv2DVariantCount; ++i) {
    const char *name = kConv2DVariants[i].name;
    if (std::strcmp(name, "Conv2DImplicitGemm") == 0)
      wide = i;
    else if (std::strcmp(name, "Conv2DImplicitGemmM256N64") == 0)
      tall = i;
    else if (std::strcmp(name, "Conv2DImplicitGemmN64") == 0)
      narrow = i;
    else if (std::strcmp(name, "Conv2DImplicitGemmS3") == 0)
      spatial = i;
  }

  // C_out is the GEMM's N, and it is what picks the tile. Measured on a 3090
  // over the cudnn_conv_bench shape table (benchmarks/op_bench.cpp sweeps all
  // three; f32 TFLOP/s, best of each row starred):
  //
  //   C_out   shape                    128x128   128x64   256x64
  //     32    56x56 3x3                  1.1       1.6*     1.0
  //     64    56x56 1x1, batch 32        8.0      12.4     12.7*
  //    128    1024px 3x3                18.0*     15.9     17.4
  //    320    sdxl-unet top block       14.6      15.7     17.0*
  //    512    vae decoder 3x3           18.6*     16.2     17.7
  //   1280    sam patch embed           15.8*     12.6     14.6
  //
  // A 128-wide tile only pays off when C_out fills it: below that its second
  // half runs masked off. 256x64 is the same 8x8 register tile in a taller,
  // 64-granular block, so it keeps full arithmetic intensity where 128x64
  // (8x4) gives a third of it away. Under 64 channels even a 64-wide tile is
  // half wasted, and the shorter 128x64 block is what keeps the grid wide
  // enough to fill the device.
  int pick;
  uint32_t pickTileM, pickTileN;
  if (C_out_ >= 128 && C_out_ % 128 == 0) {
    pick = wide;
    pickTileM = 128;
    pickTileN = 128;
  } else if (C_out_ >= 64) {
    pick = tall;
    pickTileM = 256;
    pickTileN = 64;
  } else {
    pick = narrow;
    pickTileM = 128;
    pickTileN = 64;
  }

  if (pick < 0 || !getCompiledConv2D(pick, dtype_, dtype_).has_value())
    return kConv2DDefaultVariant;

  // 3x3 stride 1 is most of the convolution in a U-Net, a VAE decoder or a
  // ResNet, and it is the one window where staging pixels beats staging im2col
  // columns: the spatial variant reuses each pixel across the three taps that
  // read it out of registers, ~1.5x less shared traffic per FMA. Measured 12-13%
  // faster than the GEMM on the shapes where neither wastes anything.
  //
  // It pays for that with a coarser grid: it rounds H_out, W_out and C_out up to
  // its patch, where the GEMM only rounds C_out and a linearised M that is far
  // too large to quantise. On 28x28 (ResNet stage 2) that rounding is 1.31x and
  // eats the win twice over, so compare padded work rather than assuming.
  if (spatial >= 0 && kH_ == 3 && kW_ == 3 && strideH_ == 1 && strideW_ == 1 &&
      getCompiledConv2D(spatial, dtype_, dtype_).has_value()) {
    const auto &s3 = kConv2DVariants[spatial];
    const auto roundUp = [](uint64_t v, uint64_t m) { return (v + m - 1) / m * m; };
    const uint64_t s3Work = static_cast<uint64_t>(N_) *
                            roundUp(H_out_, s3.effTileM) *
                            roundUp(W_out_, s3.effTileN) *
                            roundUp(C_out_, kConv2DS3TileN);
    const uint64_t m = static_cast<uint64_t>(N_) * H_out_ * W_out_;
    const uint64_t gemmWork = roundUp(m, pickTileM) * roundUp(C_out_, pickTileN);
    if (s3Work * 100 < gemmWork * 113)
      return static_cast<uint32_t>(spatial);
  }
  return static_cast<uint32_t>(pick);
}

DataType Conv2DOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> Conv2DOpNode::shader() const {
  return getCompiledConv2D(*spec_, dtype_, dtype_);
}

std::vector<uint32_t> Conv2DOpNode::outputShape() const {
  return {N_, C_out_, H_out_, W_out_};
}

ThreadSize Conv2DOpNode::dispatchSize() const {
  const auto &info = kConv2DVariants[*spec_];
  if (conv2dVariantIsSpatial3x3(static_cast<int>(*spec_))) {
    // x = tiles along W_out, y = batch * tiles along H_out, z = tiles along
    // C_out. Batch rides on y rather than z because z also carries C_out and
    // both of those are small, while y has 65535 to spend.
    const uint32_t hTiles = (H_out_ + info.effTileM - 1) / info.effTileM;
    const uint32_t gridX =
        ((W_out_ + info.effTileN - 1) / info.effTileN) * info.wgX;
    return {gridX, N_ * hTiles,
            (C_out_ + kConv2DS3TileN - 1) / kConv2DS3TileN};
  }
  if (conv2dVariantIsImplicitGemm(static_cast<int>(*spec_))) {
    // x = tiles along M = N * H_out * W_out, y = tiles along C_out. M goes on
    // x because it is the axis that grows without bound (a 2048px VAE decode
    // is 8.4M rows) and gridDim.y caps at 65535.
    const uint32_t M = N_ * H_out_ * W_out_;
    const uint32_t gridX = ((M + info.effTileM - 1) / info.effTileM) * info.wgX;
    const uint32_t gridY = (C_out_ + info.effTileN - 1) / info.effTileN;
    return {gridX, gridY, 1};
  }
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

std::vector<DataType> Conv2DOpNode::resolveInputDtypes(
    const std::vector<DataType> &inputDtypes) const {
  // Input and weight must match; if they differ, widen both
  if (inputDtypes[0] != inputDtypes[1])
    return {widenPrecision(inputDtypes[0]), widenPrecision(inputDtypes[1])};
  return inputDtypes;
}

} // namespace cut
