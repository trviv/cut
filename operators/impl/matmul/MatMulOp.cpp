#include "MatMulOp.h"
#include "TensorStore.h"

namespace cut {

// Helper: map QuantFormat to the base OperatorEnum
static OperatorEnum matmulEnum(QuantFormat fmt) {
  switch (fmt) {
  case QuantFormat::Q8:
    return MatMulQ8;
  case QuantFormat::Q4:
    return MatMulQ4;
  default:
    return MatMul;
  }
}

// Helper: get variant info fields from the format-specific table
struct VariantInfo {
  uint32_t wgX, wgY, effTileM, effTileN;
};

static VariantInfo getVariant(QuantFormat fmt, uint32_t spec) {
  switch (fmt) {
  case QuantFormat::Q8: {
    const auto &v = kMatMulQ8Variants[spec];
    return {v.wgX, v.wgY, v.effTileM, v.effTileN};
  }
  case QuantFormat::Q4: {
    const auto &v = kMatMulQ4Variants[spec];
    return {v.wgX, v.wgY, v.effTileM, v.effTileN};
  }
  default: {
    const auto &v = kMatMulVariants[spec];
    return {v.wgX, v.wgY, v.effTileM, v.effTileN};
  }
  }
}

// ============================================================================
// Standard matmul constructor (2 inputs)
// ============================================================================

MatMulOpNode::MatMulOpNode(TensorStore &store,
                           const Tensor &a,
                           const Tensor &b,
                           std::optional<uint32_t> spec)
    : OpNode(MatMul, store, spec), format_(QuantFormat::None) {
  const auto &bufA = store.getTensor(a);
  const auto &bufB = store.getTensor(b);
  const auto shapeA = bufA.getShape();
  const auto shapeB = bufB.getShape();
  dtypeA_ = bufA.getDtype();
  dtypeB_ = bufB.getDtype();
  if (shapeA.size() != 2 || shapeB.size() != 2) {
    throw std::runtime_error("matmul requires 2D matrices");
  }
  if (shapeA[1] != shapeB[0]) {
    throw std::runtime_error(
        "Matrix dimension mismatch: A is " + std::to_string(shapeA[0]) + "x" +
        std::to_string(shapeA[1]) + ", B is " + std::to_string(shapeB[0]) +
        "x" + std::to_string(shapeB[1]));
  }
  M_ = shapeA[0];
  K_ = shapeA[1];
  N_ = shapeB[1];
  if (spec.has_value()) {
    spec_ = *spec;
  } else {
    // Auto-select GEMV variant for M=1 (vector-matrix multiply).
    constexpr int kGemvVariant = kMatMulVariantCount - 1;
    spec_ = (M_ == 1) ? kGemvVariant : kMatMulDefaultVariant;
  }
  inputs_ = {a, b};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

// ============================================================================
// Quantized matmul constructor (3 inputs, auto-detect Q4/Q8)
// ============================================================================

MatMulOpNode::MatMulOpNode(TensorStore &store,
                           const Tensor &a,
                           const Tensor &packedB,
                           const Tensor &scalesB,
                           std::optional<uint32_t> spec)
    : OpNode(MatMul, store, spec), format_(QuantFormat::None) {
  const auto &bufA = store.getTensor(a);
  const auto &bufPackedB = store.getTensor(packedB);
  const auto &bufScalesB = store.getTensor(scalesB);

  const auto shapeA = bufA.getShape();
  const auto shapePB = bufPackedB.getShape();
  const auto shapeSB = bufScalesB.getShape();

  dtypeA_ = bufA.getDtype();
  dtypeB_ = bufScalesB.getDtype(); // scales dtype (Float16 typically)

  if (shapePB.size() != 2 || shapeSB.size() != 2) {
    throw std::runtime_error("matmul requires 2D B and scales matrices");
  }
  // A can be 1D [K] (treated as [1, K]) — optimizer may remove the reshape
  if (shapeA.size() == 1) {
    M_ = 1;
    K_ = shapeA[0];
  } else if (shapeA.size() == 2) {
    M_ = shapeA[0];
    K_ = shapeA[1];
  } else {
    throw std::runtime_error("matmul: A must be 1D or 2D");
  }

  // Validate B rows match K
  if (shapePB[0] != K_) {
    throw std::runtime_error("matmul: B rows (" + std::to_string(shapePB[0]) +
                             ") must match A cols (" + std::to_string(K_) +
                             ")");
  }

  // Auto-detect Q4 vs Q8 from shapes:
  //   Q8: B [K, N], scales [K/32, N] → B.cols == scales.cols
  //   Q4: B [K, N/2], scales [K/32, N] → B.cols * 2 == scales.cols
  uint32_t bCols = shapePB[1];
  uint32_t sCols = shapeSB[1];
  if (bCols == sCols) {
    format_ = QuantFormat::Q8;
    N_ = bCols;
  } else if (bCols * 2 == sCols) {
    format_ = QuantFormat::Q4;
    N_ = sCols;
  } else {
    throw std::runtime_error(
        "matmul: cannot determine quant format from B cols (" +
        std::to_string(bCols) + ") and scales cols (" + std::to_string(sCols) +
        ")");
  }

  // Set the correct OperatorEnum for this format
  op_ = matmulEnum(format_);

  if (spec.has_value()) {
    spec_ = *spec;
  } else {
    spec_ = (format_ == QuantFormat::Q8) ? kMatMulQ8DefaultVariant
                                         : kMatMulQ4DefaultVariant;
  }
  inputs_ = {a, packedB, scalesB};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

// ============================================================================
// Shared methods
// ============================================================================

DataType MatMulOpNode::outputDtype() const {
  return DataType::Float32;
}

size_t MatMulOpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(dtypeA_) & 0xF) << 16;
  key |= (static_cast<size_t>(dtypeB_) & 0xF) << 20;
  key |= static_cast<size_t>(spec_.value_or(0)) << 48;
  return key;
}

std::optional<std::vector<uint32_t>> MatMulOpNode::shader() const {
  switch (format_) {
  case QuantFormat::Q8:
    return getCompiledMatMulQ8(*spec_, dtypeA_, dtypeB_, DataType::Float32);
  case QuantFormat::Q4:
    return getCompiledMatMulQ4(*spec_, dtypeA_, dtypeB_, DataType::Float32);
  default:
    return getCompiledMatMul(*spec_, dtypeA_, dtypeB_, dtypeA_);
  }
}

std::vector<uint32_t> MatMulOpNode::outputShape() const {
  return {M_, N_};
}

ThreadSize MatMulOpNode::dispatchSize() const {
  auto vi = getVariant(format_, *spec_);
  uint32_t gridX = ((N_ + vi.effTileN - 1) / vi.effTileN) * vi.wgX;
  uint32_t gridY = ((M_ + vi.effTileM - 1) / vi.effTileM) * vi.wgY;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> MatMulOpNode::pushConstants() const {
  uint32_t strideA = (K_ + 3) & ~3u;
  if (format_ == QuantFormat::None) {
    uint32_t strideB = (N_ + 3) & ~3u;
    struct PushConstants {
      uint32_t M, K, N, strideA, strideB;
    } pc{M_, K_, N_, strideA, strideB};
    return toBytes(pc);
  }
  if (format_ == QuantFormat::Q4) {
    uint32_t strideBNpacked = (N_ / 2 + 3) & ~3u;
    uint32_t strideC = (N_ + 3) & ~3u;
    uint32_t scaleStride = (N_ + 3) & ~3u;
    struct PushConstants {
      uint32_t M, K, N, strideA, strideBNpacked, strideC, scaleStride;
    } pc{M_, K_, N_, strideA, strideBNpacked, strideC, scaleStride};
    return toBytes(pc);
  }
  // Q8
  uint32_t strideBN = (N_ + 3) & ~3u;
  uint32_t strideC = (N_ + 3) & ~3u;
  uint32_t scaleStride = (N_ + 3) & ~3u;
  struct PushConstants {
    uint32_t M, K, N, strideA, strideBN, strideC, scaleStride;
  } pc{M_, K_, N_, strideA, strideBN, strideC, scaleStride};
  return toBytes(pc);
}

std::string MatMulOpNode::displayName() const {
  switch (format_) {
  case QuantFormat::Q8:
    return "MatMulQ8";
  case QuantFormat::Q4:
    return "MatMulQ4";
  default:
    return "MatMul";
  }
}

std::vector<DataType> MatMulOpNode::resolveInputDtypes(
    const std::vector<DataType> &inputDtypes) const {
  if (format_ != QuantFormat::None) {
    // Quantized: widen A to Float32, keep packedB and scales as-is
    DataType dtA = widenPrecision(inputDtypes[0]);
    return {dtA, inputDtypes[1], inputDtypes[2]};
  }
  // Standard: try various dtype combinations
  DataType dtA = inputDtypes[0];
  DataType dtB = inputDtypes[1];
  if (getCompiledMatMul(*spec_, dtA, dtB, dtA).has_value())
    return {dtA, dtB};

  DataType wA = widenPrecision(dtA);
  DataType wB = widenPrecision(dtB);
  if (getCompiledMatMul(*spec_, dtA, wB, wB).has_value())
    return {dtA, wB};
  if (getCompiledMatMul(*spec_, wA, dtB, wA).has_value())
    return {wA, dtB};
  if (getCompiledMatMul(*spec_, wA, wB, wA).has_value())
    return {wA, wB};
  return {DataType::Float32, DataType::Float32};
}

} // namespace cut
