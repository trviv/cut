#include "MatMulQ4Op.h"
#include "MatMulQ4Variants.generated.h"
#include "TensorStore.h"

namespace cut {

MatMulQ4OpNode::MatMulQ4OpNode(TensorStore &store,
                               const Tensor &a,
                               const Tensor &packedB,
                               const Tensor &scalesB,
                               std::optional<uint32_t> spec)
    : OpNode(MatMulQ4, store, spec) {
  const auto &bufA = store.getTensor(a);
  const auto &bufPackedB = store.getTensor(packedB);
  const auto &bufScalesB = store.getTensor(scalesB);

  const auto shapeA = bufA.getShape();
  const auto shapePB = bufPackedB.getShape();
  const auto shapeSB = bufScalesB.getShape();

  dtypeA_ = bufA.getDtype();
  dtypeScales_ = bufScalesB.getDtype();

  if (shapePB.size() != 2 || shapeSB.size() != 2) {
    throw std::runtime_error("matmulQ4 requires 2D B and scales matrices");
  }
  // A can be 1D [K] (treated as [1, K]) — optimizer may remove the reshape
  if (shapeA.size() == 1) {
    M_ = 1;
    K_ = shapeA[0];
  } else if (shapeA.size() == 2) {
    M_ = shapeA[0];
    K_ = shapeA[1];
  } else {
    throw std::runtime_error("matmulQ4: A must be 1D or 2D");
  }

  // B is [K, N/2] packed nibbles; N derived from scales [K/32, N]
  N_ = shapeSB[1];

  // Validate: first dim of B should match K
  if (shapePB[0] != K_) {
    throw std::runtime_error("matmulQ4: B rows (" + std::to_string(shapePB[0]) +
                             ") must match A cols (" + std::to_string(K_) +
                             ")");
  }
  // Validate: second dim of B should be N/2
  if (shapePB[1] != N_ / 2) {
    throw std::runtime_error("matmulQ4: B cols (" + std::to_string(shapePB[1]) +
                             ") must be N/2 (" + std::to_string(N_ / 2) + ")");
  }

  if (spec.has_value()) {
    spec_ = *spec;
  } else {
    spec_ = kMatMulQ4DefaultVariant;
  }
  inputs_ = {a, packedB, scalesB};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType MatMulQ4OpNode::outputDtype() const {
  return DataType::Float32;
}

size_t MatMulQ4OpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(dtypeA_) & 0xF) << 16;
  key |= (static_cast<size_t>(dtypeScales_) & 0xF) << 20;
  key |= static_cast<size_t>(spec_.value_or(0)) << 48;
  return key;
}

std::optional<std::vector<uint32_t>> MatMulQ4OpNode::shader() const {
  return getCompiledMatMulQ4(*spec_, dtypeA_, dtypeScales_, DataType::Float32);
}

std::vector<uint32_t> MatMulQ4OpNode::outputShape() const {
  return {M_, N_};
}

ThreadSize MatMulQ4OpNode::dispatchSize() const {
  const auto &info = kMatMulQ4Variants[*spec_];
  uint32_t gridX = ((N_ + info.effTileN - 1) / info.effTileN) * info.wgX;
  uint32_t gridY = ((M_ + info.effTileM - 1) / info.effTileM) * info.wgY;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> MatMulQ4OpNode::pushConstants() const {
  uint32_t strideA = (K_ + 3) & ~3u;
  uint32_t strideBNpacked = (N_ / 2 + 3) & ~3u; // B is [K, N/2] packed
  uint32_t strideC = (N_ + 3) & ~3u;
  uint32_t scaleStride = (N_ + 3) & ~3u; // scales [K/32, N], inner dim is N
  struct PushConstants {
    uint32_t M, K, N, strideA, strideBNpacked, strideC, scaleStride;
  } pc{M_, K_, N_, strideA, strideBNpacked, strideC, scaleStride};
  return toBytes(pc);
}

std::string MatMulQ4OpNode::displayName() const {
  return "MatMulQ4";
}

std::vector<DataType> MatMulQ4OpNode::resolveInputDtypes(
    const std::vector<DataType> &inputDtypes) const {
  DataType dtA = widenPrecision(inputDtypes[0]);
  return {dtA, inputDtypes[1], inputDtypes[2]};
}

} // namespace cut
