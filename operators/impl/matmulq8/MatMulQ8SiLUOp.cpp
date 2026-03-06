#include "MatMulQ8SiLUOp.h"
#include "MatMulQ8Variants.generated.h"
#include "TensorStore.h"

namespace cut {

MatMulQ8SiLUOpNode::MatMulQ8SiLUOpNode(TensorStore &store,
                                       const Tensor &a,
                                       const Tensor &packedB,
                                       const Tensor &scalesB,
                                       uint32_t bCols,
                                       std::optional<uint32_t> spec)
    : OpNode(MatMulQ8SiLU, store, spec) {
  const auto &bufA = store.getTensor(a);
  const auto &bufPackedB = store.getTensor(packedB);
  const auto &bufScalesB = store.getTensor(scalesB);

  const auto shapeA = bufA.getShape();
  const auto shapePB = bufPackedB.getShape();
  const auto shapeSB = bufScalesB.getShape();

  dtypeA_ = bufA.getDtype();
  dtypeScales_ = bufScalesB.getDtype();

  if (shapePB.size() != 2 || shapeSB.size() != 2) {
    throw std::runtime_error("matmulQ8SiLU requires 2D B and scales matrices");
  }
  // A can be 1D [K] (treated as [1, K]) — optimizer may remove the reshape
  if (shapeA.size() == 1) {
    M_ = 1;
    K_ = shapeA[0];
  } else if (shapeA.size() == 2) {
    M_ = shapeA[0];
    K_ = shapeA[1];
  } else {
    throw std::runtime_error("matmulQ8SiLU: A must be 1D or 2D");
  }
  N_ = shapePB[1]; // B is [K, N] (transposed at load time)

  if (shapePB[0] != K_) {
    throw std::runtime_error(
        "matmulQ8SiLU: B rows (" + std::to_string(shapePB[0]) +
        ") must match A cols (" + std::to_string(K_) + ")");
  }

  if (spec.has_value()) {
    spec_ = *spec;
  } else {
    constexpr uint32_t kSiLUTiledVariant = 1; // SiLUT16R4x4
    spec_ = kSiLUTiledVariant;
  }
  inputs_ = {a, packedB, scalesB};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType MatMulQ8SiLUOpNode::outputDtype() const {
  return DataType::Float32;
}

size_t MatMulQ8SiLUOpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(dtypeA_) & 0xF) << 16;
  key |= (static_cast<size_t>(dtypeScales_) & 0xF) << 20;
  key |= static_cast<size_t>(spec_.value_or(0)) << 48;
  return key;
}

std::optional<std::vector<uint32_t>> MatMulQ8SiLUOpNode::shader() const {
  return getCompiledMatMulQ8(*spec_, dtypeA_, dtypeScales_, DataType::Float32);
}

std::vector<uint32_t> MatMulQ8SiLUOpNode::outputShape() const {
  return {M_, N_};
}

ThreadSize MatMulQ8SiLUOpNode::dispatchSize() const {
  const auto &info = kMatMulQ8Variants[*spec_];
  uint32_t gridX = ((N_ + info.effTileN - 1) / info.effTileN) * info.wgX;
  uint32_t gridY = ((M_ + info.effTileM - 1) / info.effTileM) * info.wgY;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> MatMulQ8SiLUOpNode::pushConstants() const {
  uint32_t strideA = (K_ + 3) & ~3u;
  uint32_t strideBN = (N_ + 3) & ~3u;
  uint32_t strideC = (N_ + 3) & ~3u;
  uint32_t scaleStride = (N_ + 3) & ~3u;
  struct PushConstants {
    uint32_t M, K, N, strideA, strideBN, strideC, scaleStride;
  } pc{M_, K_, N_, strideA, strideBN, strideC, scaleStride};
  return toBytes(pc);
}

std::string MatMulQ8SiLUOpNode::displayName() const {
  return "MatMulQ8SiLU";
}

std::vector<DataType> MatMulQ8SiLUOpNode::resolveInputDtypes(
    const std::vector<DataType> &inputDtypes) const {
  DataType dtA = widenPrecision(inputDtypes[0]);
  return {dtA, inputDtypes[1], inputDtypes[2]};
}

} // namespace cut
