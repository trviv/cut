#include "MatMulBinaryOp.h"
#include "Shaders.h"
#include "TensorStore.h"

namespace cut {

static OperatorEnum binaryEnum(QuantFormat fmt) {
  return (fmt == QuantFormat::Q4) ? MatMulQ4Binary : MatMulQ8Binary;
}

struct BinaryVariantInfo {
  uint32_t wgX, wgY, effTileM, effTileN;
};

static BinaryVariantInfo getBinaryVariant(QuantFormat fmt, uint32_t spec) {
  switch (fmt) {
  case QuantFormat::Q4: {
    const auto &v = kMatMulQ4Variants[spec];
    return {v.wgX, v.wgY, v.effTileM, v.effTileN};
  }
  default: {
    const auto &v = kMatMulQ8Variants[spec];
    return {v.wgX, v.wgY, v.effTileM, v.effTileN};
  }
  }
}

MatMulBinaryOpNode::MatMulBinaryOpNode(TensorStore &store,
                                       OperatorEnum binaryOp,
                                       const Tensor &a,
                                       const Tensor &packedB,
                                       const Tensor &scalesB,
                                       const Tensor &d,
                                       std::optional<uint32_t> spec)
    : OpNode(MatMulQ8Binary, store, spec), binaryOp_(binaryOp),
      format_(QuantFormat::Q8) {
  const auto &bufA = store.getTensor(a);
  const auto &bufPackedB = store.getTensor(packedB);
  const auto &bufScalesB = store.getTensor(scalesB);
  const auto &bufD = store.getTensor(d);

  const auto shapeA = bufA.getShape();
  const auto shapePB = bufPackedB.getShape();
  const auto shapeSB = bufScalesB.getShape();
  const auto shapeD = bufD.getShape();

  dtypeA_ = bufA.getDtype();
  dtypeScales_ = bufScalesB.getDtype();

  if (shapePB.size() != 2 || shapeSB.size() != 2) {
    throw std::runtime_error("matmulBinary requires 2D B and scales matrices");
  }
  if (shapeA.size() == 1) {
    M_ = 1;
    K_ = shapeA[0];
  } else if (shapeA.size() == 2) {
    M_ = shapeA[0];
    K_ = shapeA[1];
  } else {
    throw std::runtime_error("matmulBinary: A must be 1D or 2D");
  }

  if (shapePB[0] != K_) {
    throw std::runtime_error(
        "matmulBinary: B rows (" + std::to_string(shapePB[0]) +
        ") must match A cols (" + std::to_string(K_) + ")");
  }

  // Auto-detect Q4 vs Q8
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
        "matmulBinary: cannot determine quant format from B cols (" +
        std::to_string(bCols) + ") and scales cols (" + std::to_string(sCols) +
        ")");
  }

  op_ = binaryEnum(format_);

  // Validate D shape matches output [M, N]
  size_t expectedElements = static_cast<size_t>(M_) * N_;
  if (actualElementCount(shapeD) != expectedElements) {
    throw std::runtime_error(
        "matmulBinary: D element count (" +
        std::to_string(actualElementCount(shapeD)) +
        ") must match output [M,N] = " + std::to_string(expectedElements));
  }

  constexpr uint32_t kBinaryTiledVariant = 4; // BinaryT16R4x4
  spec_ = spec.value_or(kBinaryTiledVariant);

  inputs_ = {a, packedB, scalesB, d};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType MatMulBinaryOpNode::outputDtype() const {
  return DataType::Float32;
}

size_t MatMulBinaryOpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(dtypeA_) & 0xF) << 16;
  key |= (static_cast<size_t>(dtypeScales_) & 0xF) << 20;
  key |= (static_cast<size_t>(binaryOp_) & 0xFF) << 24;
  key |= static_cast<size_t>(spec_.value_or(0)) << 48;
  return key;
}

std::optional<std::vector<uint32_t>> MatMulBinaryOpNode::shader() const {
  std::optional<std::vector<uint32_t>> compiled;
  if (format_ == QuantFormat::Q4) {
    compiled =
        getCompiledMatMulQ4(*spec_, dtypeA_, dtypeScales_, DataType::Float32);
  } else {
    compiled =
        getCompiledMatMulQ8(*spec_, dtypeA_, dtypeScales_, DataType::Float32);
  }
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(binaryOp_));
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> MatMulBinaryOpNode::outputShape() const {
  return {M_, N_};
}

ThreadSize MatMulBinaryOpNode::dispatchSize() const {
  auto vi = getBinaryVariant(format_, *spec_);
  uint32_t gridX = ((N_ + vi.effTileN - 1) / vi.effTileN) * vi.wgX;
  uint32_t gridY = ((M_ + vi.effTileM - 1) / vi.effTileM) * vi.wgY;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> MatMulBinaryOpNode::pushConstants() const {
  uint32_t strideA = (K_ + 3) & ~3u;
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

std::string MatMulBinaryOpNode::displayName() const {
  return (format_ == QuantFormat::Q4) ? "MatMulQ4Binary" : "MatMulQ8Binary";
}

std::vector<DataType> MatMulBinaryOpNode::resolveInputDtypes(
    const std::vector<DataType> &inputDtypes) const {
  DataType dtA = widenPrecision(inputDtypes[0]);
  DataType dtD = widenPrecision(inputDtypes[3]);
  return {dtA, inputDtypes[1], inputDtypes[2], dtD};
}

} // namespace cut
