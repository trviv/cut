#include "MatMulSiLUOp.h"
#include "MatMulVariants.generated.h"
#include "TensorStore.h"

namespace cut {

MatMulSiLUOpNode::MatMulSiLUOpNode(TensorStore &store,
                                   const Tensor &a,
                                   const Tensor &b,
                                   std::optional<uint32_t> spec)
    : OpNode(MatMulSiLU, store, spec) {
  const auto &bufA = store.getTensor(a);
  const auto &bufB = store.getTensor(b);
  const auto shapeA = bufA.getShape();
  const auto shapeB = bufB.getShape();
  dtypeA_ = bufA.getDtype();
  dtypeB_ = bufB.getDtype();

  if (shapeA.size() != 2 || shapeB.size() != 2) {
    throw std::runtime_error("matmulSiLU requires 2D matrices");
  }
  if (shapeA[1] != shapeB[0]) {
    throw std::runtime_error("Matrix dimension mismatch in matmulSiLU: A is " +
                             std::to_string(shapeA[0]) + "x" +
                             std::to_string(shapeA[1]) + ", B is " +
                             std::to_string(shapeB[0]) + "x" +
                             std::to_string(shapeB[1]));
  }

  M_ = shapeA[0];
  K_ = shapeA[1];
  N_ = shapeB[1];

  // Use SiLUT16R4x4 variant (index 18 in MatMul variants table).
  // This variant includes inline SiLU activation.
  constexpr uint32_t kMatMulSiLUVariant = 18;
  spec_ = spec.value_or(kMatMulSiLUVariant);

  inputs_ = {a, b};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType MatMulSiLUOpNode::outputDtype() const {
  return DataType::Float32;
}

size_t MatMulSiLUOpNode::shaderKey() const {
  // Override to encode dtypeA_ and dtypeB_ in distinct slots
  // Similar to MatMulOpNode::shaderKey()
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(dtypeA_) & 0xF) << 16;
  key |= (static_cast<size_t>(dtypeB_) & 0xF) << 20;
  key |= static_cast<size_t>(spec_.value_or(0)) << 48;
  return key;
}

std::optional<std::vector<uint32_t>> MatMulSiLUOpNode::shader() const {
  // Use MatMul variant infrastructure (SiLUT16R4x4 is part of MatMul variants)
  return getCompiledMatMul(*spec_, dtypeA_, dtypeB_, dtypeA_);
}

std::vector<uint32_t> MatMulSiLUOpNode::outputShape() const {
  return {M_, N_};
}

ThreadSize MatMulSiLUOpNode::dispatchSize() const {
  // Use same dispatch logic as MatMul
  const auto &info = kMatMulVariants[*spec_];
  uint32_t gridX = ((N_ + info.effTileN - 1) / info.effTileN) * info.wgX;
  uint32_t gridY = ((M_ + info.effTileM - 1) / info.effTileM) * info.wgY;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> MatMulSiLUOpNode::pushConstants() const {
  // Same push constants as MatMul
  uint32_t strideA = (K_ + 3) & ~3u;
  uint32_t strideB = (N_ + 3) & ~3u;
  struct PushConstants {
    uint32_t M, K, N, strideA, strideB;
  } pc{M_, K_, N_, strideA, strideB};
  return toBytes(pc);
}

std::vector<DataType> MatMulSiLUOpNode::resolveInputDtypes(
    const std::vector<DataType> &inputDtypes) const {
  DataType dtA = inputDtypes[0];
  DataType dtB = inputDtypes[1];

  // For now, use same resolution logic as MatMul
  // Check if (dtA, dtB) is directly supported
  if (getCompiledMatMul(*spec_, dtA, dtB, dtA).has_value())
    return {dtA, dtB};

  // Try widening both to higher precision
  DataType wA = widenPrecision(dtA);
  DataType wB = widenPrecision(dtB);
  if (getCompiledMatMul(*spec_, wA, wB, wA).has_value())
    return {wA, wB};

  // Try mixed: keep A, widen B
  if (getCompiledMatMul(*spec_, dtA, wB, dtA).has_value())
    return {dtA, wB};

  // Try mixed: widen A, keep B
  if (getCompiledMatMul(*spec_, wA, dtB, wA).has_value())
    return {wA, dtB};

  // Fallback: both Float32
  return {DataType::Float32, DataType::Float32};
}

} // namespace cut
