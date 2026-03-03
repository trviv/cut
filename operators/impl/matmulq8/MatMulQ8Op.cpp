#include "MatMulQ8Op.h"
#include "MatMulQ8Variants.generated.h"
#include "TensorStore.h"

namespace cut {

MatMulQ8OpNode::MatMulQ8OpNode(TensorStore &store,
                               const Tensor &a,
                               const Tensor &packedB,
                               const Tensor &scalesB,
                               uint32_t bCols,
                               std::optional<uint32_t> spec)
    : OpNode(MatMulQ8, store, spec) {
  const auto &bufA = store.getTensor(a);
  const auto &bufPackedB = store.getTensor(packedB);
  const auto &bufScalesB = store.getTensor(scalesB);

  const auto shapeA = bufA.getShape();
  const auto shapePB = bufPackedB.getShape();
  const auto shapeSB = bufScalesB.getShape();

  dtypeA_ = bufA.getDtype();

  if (shapeA.size() != 2 || shapePB.size() != 2 || shapeSB.size() != 2) {
    throw std::runtime_error("matmulQ8 requires 2D matrices");
  }

  M_ = shapeA[0];
  K_ = shapeA[1];
  N_ = shapePB[0]; // B_original is [N, K] in GGUF layout

  // Validate: bCols should match K
  if (bCols != K_) {
    throw std::runtime_error("matmulQ8: bCols (" + std::to_string(bCols) +
                             ") must match A cols (" + std::to_string(K_) +
                             ")");
  }

  spec_ = spec.value_or(kMatMulQ8DefaultVariant);
  inputs_ = {a, packedB, scalesB};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType MatMulQ8OpNode::outputDtype() const {
  return DataType::Float32;
}

size_t MatMulQ8OpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(dtypeA_) & 0xF) << 16;
  key |= static_cast<size_t>(spec_.value_or(0)) << 48;
  return key;
}

std::optional<std::vector<uint32_t>> MatMulQ8OpNode::shader() const {
  return getCompiledMatMulQ8(*spec_, dtypeA_, DataType::Float32);
}

std::vector<uint32_t> MatMulQ8OpNode::outputShape() const {
  return {M_, N_};
}

ThreadSize MatMulQ8OpNode::dispatchSize() const {
  const auto &info = kMatMulQ8Variants[*spec_];
  uint32_t gridX = ((N_ + info.effTileN - 1) / info.effTileN) * info.wgX;
  uint32_t gridY = ((M_ + info.effTileM - 1) / info.effTileM) * info.wgY;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> MatMulQ8OpNode::pushConstants() const {
  uint32_t strideA = (K_ + 3) & ~3u;
  uint32_t strideBK = (K_ + 3) & ~3u;
  uint32_t strideC = (N_ + 3) & ~3u;
  uint32_t blocksPerRow = (K_ + 31) / 32;
  uint32_t scaleStride = (blocksPerRow + 3) & ~3u;
  struct PushConstants {
    uint32_t M, K, N, strideA, strideBK, strideC, scaleStride;
  } pc{M_, K_, N_, strideA, strideBK, strideC, scaleStride};
  return toBytes(pc);
}

std::string MatMulQ8OpNode::displayName() const {
  return "MatMulQ8";
}

std::vector<DataType> MatMulQ8OpNode::resolveInputDtypes(
    const std::vector<DataType> &inputDtypes) const {
  // Input 0 (A): widen to Float32 if needed.
  // Inputs 1,2 (packedB, scalesB): keep as-is (Int8 and Float16).
  DataType dtA = widenPrecision(inputDtypes[0]);
  return {dtA, inputDtypes[1], inputDtypes[2]};
}

} // namespace cut
