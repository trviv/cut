#include "MatMulOp.h"
#include "Shaders.h"
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
  //   Q8: B [K, N], scales [K/32, N] -> B.cols == scales.cols
  //   Q4: B [K, N/2], scales [K/32, N] -> B.cols * 2 == scales.cols
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
// Fusion support
// ============================================================================

void MatMulOpNode::setFusion(MatMulFusion fusion,
                             OperatorEnum fusionOp,
                             const Tensor &d) {
  fusion_ = fusion;
  fusionOp_ = fusionOp;
  if (fusion == MatMulFusion::Binary) {
    inputs_.push_back(d);
  }
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
  // Encode fusion type (bits 36-39) and binary op (bits 40-47)
  key |= (static_cast<size_t>(fusion_) & 0xF) << 36;
  if (fusion_ != MatMulFusion::None) {
    key |= (static_cast<size_t>(fusionOp_) & 0xFF) << 40;
  }
  key |= static_cast<size_t>(spec_.value_or(0)) << 48;
  return key;
}

std::optional<std::vector<uint32_t>> MatMulOpNode::shader() const {
  // Get base matmul SPIR-V
  std::optional<std::vector<uint32_t>> compiled;
  switch (format_) {
  case QuantFormat::Q8:
    compiled = getCompiledMatMulQ8(*spec_, dtypeA_, dtypeB_, DataType::Float32);
    break;
  case QuantFormat::Q4:
    compiled = getCompiledMatMulQ4(*spec_, dtypeA_, dtypeB_, DataType::Float32);
    break;
  default:
    compiled = getCompiledMatMul(*spec_, dtypeA_, dtypeB_, dtypeA_);
    break;
  }

  if (!compiled.has_value() || fusion_ == MatMulFusion::None) {
    return compiled;
  }

  auto spirv = std::move(compiled.value());

  // Patch specialization constants for fusion (single SPIR-V scan)
  if (fusion_ == MatMulFusion::Unary) {
    patchSpecConstants(spirv, {{1, 1}, {2, static_cast<uint32_t>(fusionOp_)}});
  } else if (fusion_ == MatMulFusion::Binary) {
    patchSpecConstants(spirv, {{1, 2}, {2, static_cast<uint32_t>(fusionOp_)}});
  }

  return spirv;
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
    uint32_t strideC = strideB;
    struct PushConstants {
      uint32_t M, K, N, strideA, strideB, strideC;
    } pc{M_, K_, N_, strideA, strideB, strideC};
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

std::vector<ComputeBinding> MatMulOpNode::bindings() const {
  std::vector<ComputeBinding> result;
  uint32_t idx = 0;

  // Bind all inputs (A, B, [scalesB], [D for Binary fusion])
  for (const auto &h : inputs_) {
    result.emplace_back(idx++, h);
  }

  // For non-Binary fusion, insert a dummy D binding (use output_ as dummy).
  // Binary fusion already has D as the last entry in inputs_.
  if (fusion_ != MatMulFusion::Binary) {
    result.emplace_back(idx++, output_);
  }

  // Bind output C
  if (output_) {
    result.emplace_back(idx++, output_);
  }

  // Push constants
  auto pc = pushConstants();
  if (!pc.empty()) {
    result.emplace_back(
        idx, DataReference(pc.data(), static_cast<uint32_t>(pc.size())));
  }
  return result;
}

std::string MatMulOpNode::displayName() const {
  std::string name;
  switch (format_) {
  case QuantFormat::Q8:
    name = "MatMulQ8";
    break;
  case QuantFormat::Q4:
    name = "MatMulQ4";
    break;
  default:
    name = "MatMul";
    break;
  }
  switch (fusion_) {
  case MatMulFusion::Unary:
    name += "Unary";
    break;
  case MatMulFusion::Binary:
    name += "Binary";
    break;
  default:
    break;
  }
  return name;
}

std::vector<DataType> MatMulOpNode::resolveInputDtypes(
    const std::vector<DataType> &inputDtypes) const {
  if (format_ != QuantFormat::None) {
    // Quantized: widen A to Float32, keep packedB and scales as-is
    DataType dtA = widenPrecision(inputDtypes[0]);
    std::vector<DataType> result = {dtA, inputDtypes[1], inputDtypes[2]};
    // Binary fusion has D as 4th input
    if (fusion_ == MatMulFusion::Binary && inputDtypes.size() > 3) {
      result.push_back(widenPrecision(inputDtypes[3]));
    }
    return result;
  }
  // Standard: try various dtype combinations
  DataType dtA = inputDtypes[0];
  DataType dtB = inputDtypes[1];

  std::vector<DataType> result;
  if (getCompiledMatMul(*spec_, dtA, dtB, dtA).has_value())
    result = {dtA, dtB};
  else {
    DataType wA = widenPrecision(dtA);
    DataType wB = widenPrecision(dtB);
    if (getCompiledMatMul(*spec_, dtA, wB, wB).has_value())
      result = {dtA, wB};
    else if (getCompiledMatMul(*spec_, wA, dtB, wA).has_value())
      result = {wA, dtB};
    else if (getCompiledMatMul(*spec_, wA, wB, wA).has_value())
      result = {wA, wB};
    else
      result = {DataType::Float32, DataType::Float32};
  }

  // Binary fusion has D as 3rd input for standard format
  if (fusion_ == MatMulFusion::Binary && inputDtypes.size() > 2) {
    result.push_back(widenPrecision(inputDtypes[2]));
  }
  return result;
}

} // namespace cut
