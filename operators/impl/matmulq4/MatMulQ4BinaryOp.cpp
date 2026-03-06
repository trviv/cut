#include "MatMulQ4BinaryOp.h"
#include "MatMulQ4Variants.generated.h"
#include "Shaders.h"
#include "TensorStore.h"

namespace cut {

MatMulQ4BinaryOpNode::MatMulQ4BinaryOpNode(TensorStore &store,
                                           OperatorEnum binaryOp,
                                           const Tensor &a,
                                           const Tensor &packedB,
                                           const Tensor &scalesB,
                                           const Tensor &d,
                                           uint32_t /*bCols*/,
                                           std::optional<uint32_t> spec)
    : OpNode(MatMulQ4Binary, store, spec), binaryOp_(binaryOp) {
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
    throw std::runtime_error(
        "matmulQ4Binary requires 2D B and scales matrices");
  }
  if (shapeA.size() == 1) {
    M_ = 1;
    K_ = shapeA[0];
  } else if (shapeA.size() == 2) {
    M_ = shapeA[0];
    K_ = shapeA[1];
  } else {
    throw std::runtime_error("matmulQ4Binary: A must be 1D or 2D");
  }
  N_ = shapeSB[1]; // N from scales [K/32, N]

  if (shapePB[0] != K_) {
    throw std::runtime_error(
        "matmulQ4Binary: B rows (" + std::to_string(shapePB[0]) +
        ") must match A cols (" + std::to_string(K_) + ")");
  }

  // Validate D shape matches output [M, N]
  size_t expectedElements = static_cast<size_t>(M_) * N_;
  if (actualElementCount(shapeD) != expectedElements) {
    throw std::runtime_error(
        "matmulQ4Binary: D element count (" +
        std::to_string(actualElementCount(shapeD)) +
        ") must match output [M,N] = " + std::to_string(expectedElements));
  }

  if (spec.has_value()) {
    spec_ = *spec;
  } else {
    constexpr uint32_t kBinaryTiledVariant = 4; // BinaryT16R4x4
    spec_ = kBinaryTiledVariant;
  }
  inputs_ = {a, packedB, scalesB, d};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType MatMulQ4BinaryOpNode::outputDtype() const {
  return DataType::Float32;
}

size_t MatMulQ4BinaryOpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(dtypeA_) & 0xF) << 16;
  key |= (static_cast<size_t>(dtypeScales_) & 0xF) << 20;
  key |= (static_cast<size_t>(binaryOp_) & 0xFF) << 24;
  key |= static_cast<size_t>(spec_.value_or(0)) << 48;
  return key;
}

std::optional<std::vector<uint32_t>> MatMulQ4BinaryOpNode::shader() const {
  auto compiled =
      getCompiledMatMulQ4(*spec_, dtypeA_, dtypeScales_, DataType::Float32);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(binaryOp_));
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> MatMulQ4BinaryOpNode::outputShape() const {
  return {M_, N_};
}

ThreadSize MatMulQ4BinaryOpNode::dispatchSize() const {
  const auto &info = kMatMulQ4Variants[*spec_];
  uint32_t gridX = ((N_ + info.effTileN - 1) / info.effTileN) * info.wgX;
  uint32_t gridY = ((M_ + info.effTileM - 1) / info.effTileM) * info.wgY;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> MatMulQ4BinaryOpNode::pushConstants() const {
  uint32_t strideA = (K_ + 3) & ~3u;
  uint32_t strideBNpacked = (N_ / 2 + 3) & ~3u;
  uint32_t strideC = (N_ + 3) & ~3u;
  uint32_t scaleStride = (N_ + 3) & ~3u;
  struct PushConstants {
    uint32_t M, K, N, strideA, strideBNpacked, strideC, scaleStride;
  } pc{M_, K_, N_, strideA, strideBNpacked, strideC, scaleStride};
  return toBytes(pc);
}

std::string MatMulQ4BinaryOpNode::displayName() const {
  return "MatMulQ4Binary";
}

std::vector<DataType> MatMulQ4BinaryOpNode::resolveInputDtypes(
    const std::vector<DataType> &inputDtypes) const {
  DataType dtA = widenPrecision(inputDtypes[0]);
  DataType dtD = widenPrecision(inputDtypes[3]);
  return {dtA, inputDtypes[1], inputDtypes[2], dtD};
}

} // namespace cut
