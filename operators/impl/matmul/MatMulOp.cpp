#include "MatMulOp.h"
#include "Runtime.h"

namespace cut {

MatMulOpNode::MatMulOpNode(Runtime &runtime,
                           const Tensor &a,
                           const Tensor &b,
                           std::optional<uint32_t> spec)
    : OpNode(MatMul, runtime, spec) {
  const auto &bufA = runtime.getTensor(a);
  const auto &bufB = runtime.getTensor(b);
  const auto shapeA = bufA.getShape();
  const auto shapeB = bufB.getShape();
  dtype_ = bufA.getDtype();
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
  resolvedVariant_ = spec.value_or(kMatMulDefaultVariant);
  inputs_ = {a, b};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType MatMulOpNode::shaderDtype() const {
  return dtype_;
}
DataType MatMulOpNode::outputDtype() const {
  return DataType::Float32;
}
std::optional<uint32_t> MatMulOpNode::spec() const {
  return resolvedVariant_;
}

std::optional<std::vector<uint32_t>> MatMulOpNode::shader() const {
  return getCompiledMatMul(resolvedVariant_, dtype_);
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
