#pragma once

#include "OpNode.h"
#include "impl/conv2d/Conv2DVariants.generated.h"

#include <cstring>

namespace cut {

/// True for the implicit-GEMM family, which tiles the (M = N*H_out*W_out,
/// C_out) output plane rather than the (W_out, H_out, N*C_out) one the direct
/// variants use — so it needs its own dispatch geometry.
///
/// Keyed on the variant name rather than a manifest field for the same reason
/// the scan family does it: adding a field to generate_shader_variants.py
/// rewrites every operator's *Variants.generated.h.
inline bool conv2dVariantIsImplicitGemm(int variantIndex) {
  if (variantIndex < 0 || variantIndex >= kConv2DVariantCount)
    return false;
  return std::strstr(kConv2DVariants[variantIndex].name, "ImplicitGemm") !=
         nullptr;
}

/// Output channels per block in the spatially-tiled 3x3 variant. The other two
/// tile extents travel in the variant table as eff_tile = (S3_TH, S3_TW); this
/// one has nowhere to ride along, so it is mirrored here and must stay equal to
/// S3_BN in Conv2DImplicitGemmS3.{cu,shader}.
inline constexpr uint32_t kConv2DS3TileN = 128;

/// True for the spatially-tiled 3x3 stride-1 variant, which stages input pixels
/// instead of im2col columns and so tiles (H_out, W_out, C_out) rather than
/// (M, C_out).
inline bool conv2dVariantIsSpatial3x3(int variantIndex) {
  if (variantIndex < 0 || variantIndex >= kConv2DVariantCount)
    return false;
  return std::strcmp(kConv2DVariants[variantIndex].name,
                     "Conv2DImplicitGemmS3") == 0;
}

/// False for a (variant, window) pair the variant cannot express.
///
/// The direct tiled variant stages (TILE-1)*stride + k input pixels per axis in
/// a shared tile sized for stride <= 2 and k <= 11, and reads past that tile
/// rather than failing when the window is larger — a stride-16 patch embedding
/// walks 256 rows into a 42-row tile. Callers pin variants by index (autotune,
/// op_bench, the variant tests), so the guard belongs here rather than in each
/// of them.
inline bool conv2dVariantSupportsShape(int variantIndex,
                                       uint32_t kH,
                                       uint32_t kW,
                                       uint32_t strideH,
                                       uint32_t strideW) {
  if (variantIndex < 0 || variantIndex >= kConv2DVariantCount)
    return false;
  const auto &info = kConv2DVariants[variantIndex];
  // The sliding window the spatial variant reuses out of registers only
  // overlaps at unit stride, and its patch is sized for a 3x3.
  if (conv2dVariantIsSpatial3x3(variantIndex))
    return kH == 3 && kW == 3 && strideH == 1 && strideW == 1;
  if (std::strcmp(info.name, "Conv2DTiled") != 0)
    return true;
  // Mirrors SHARED_H/SHARED_W in Conv2DTiled.{cu,shader}: TILE * 2 + MAX_K - 1.
  return (info.effTileM - 1) * strideH + kH <= info.effTileM * 2 + 10 &&
         (info.effTileN - 1) * strideW + kW <= info.effTileN * 2 + 10;
}

class Conv2DOpNode : public OpNode {
public:
  Conv2DOpNode(TensorStore &store,
               const Tensor &input,
               const Tensor &weight,
               uint32_t strideH,
               uint32_t strideW,
               uint32_t padH,
               uint32_t padW,
               std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::vector<DataType>
  resolveInputDtypes(const std::vector<DataType> &inputDtypes) const override;

private:
  /// Variant used when the caller did not pin one: backend- and shape-aware.
  uint32_t defaultVariant(TensorStore &store) const;

  uint32_t strideH_, strideW_, padH_, padW_;
  DataType dtype_;
  uint32_t N_, C_in_, H_in_, W_in_, C_out_, kH_, kW_, H_out_, W_out_;
};

} // namespace cut
