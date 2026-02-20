#include "MatMulOp.h"

namespace cut {

MatMulOpNode::MatMulOpNode(std::vector<uint32_t> shapeA,
                           std::vector<uint32_t> shapeB,
                           DataType dtype,
                           int variantIdx)
    : shapeA_(std::move(shapeA)), shapeB_(std::move(shapeB)), dtype_(dtype) {
  if (shapeA_.size() != 2 || shapeB_.size() != 2) {
    throw std::runtime_error("matmul requires 2D matrices");
  }
  if (shapeA_[1] != shapeB_[0]) {
    throw std::runtime_error(
        "Matrix dimension mismatch: A is " + std::to_string(shapeA_[0]) + "x" +
        std::to_string(shapeA_[1]) + ", B is " + std::to_string(shapeB_[0]) +
        "x" + std::to_string(shapeB_[1]));
  }
  M_ = shapeA_[0];
  K_ = shapeA_[1];
  N_ = shapeB_[1];
  resolvedVariant_ = variantIdx >= 0 ? variantIdx : kMatMulDefaultVariant;
}

OperatorEnum MatMulOpNode::op() const {
  return MatMul;
}
DataType MatMulOpNode::shaderDtype() const {
  return dtype_;
}
DataType MatMulOpNode::outputDtype() const {
  return DataType::Float32;
}
int MatMulOpNode::variantIndex() const {
  return resolvedVariant_;
}

std::vector<uint32_t> MatMulOpNode::outputShape() const {
  return {M_, N_};
}

ThreadSize MatMulOpNode::dispatchSize() const {
  const auto &info = kMatMulVariants[resolvedVariant_];
  uint32_t gridX = ((N_ + info.effTileN - 1) / info.effTileN) * info.wgX;
  uint32_t gridY = ((M_ + info.effTileM - 1) / info.effTileM) * info.wgY;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> MatMulOpNode::pushConstants() const {
  uint32_t strideA = (K_ + 3) & ~3u;
  uint32_t strideB = (N_ + 3) & ~3u;
  struct PushConstants {
    uint32_t M, K, N, strideA, strideB;
  } pc{M_, K_, N_, strideA, strideB};
  return toBytes(pc);
}

} // namespace cut
