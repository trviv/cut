#pragma once

#include "OpNode.h"
#include "impl/matmul/MatMulVariants.generated.h"

namespace cut {

class MatMulOpNode : public OpNode {
public:
  MatMulOpNode(std::vector<uint32_t> shapeA,
               std::vector<uint32_t> shapeB,
               DataType dtype,
               int variantIdx = -1)
      : shapeA_(std::move(shapeA)), shapeB_(std::move(shapeB)), dtype_(dtype) {
    M_ = shapeA_[0];
    K_ = shapeA_[1];
    N_ = shapeB_[1];
    resolvedVariant_ = variantIdx >= 0 ? variantIdx : kMatMulDefaultVariant;
  }

  void validate() const override {
    if (shapeA_.size() != 2 || shapeB_.size() != 2) {
      throw std::runtime_error("matmul requires 2D matrices");
    }
    if (shapeA_[1] != shapeB_[0]) {
      throw std::runtime_error(
          "Matrix dimension mismatch: A is " + std::to_string(shapeA_[0]) +
          "x" + std::to_string(shapeA_[1]) + ", B is " +
          std::to_string(shapeB_[0]) + "x" + std::to_string(shapeB_[1]));
    }
  }

  OperatorEnum op() const override { return MatMul; }
  DataType shaderDtype() const override { return dtype_; }
  DataType outputDtype() const override { return DataType::Float32; }
  int variantIndex() const override { return resolvedVariant_; }

  std::vector<uint32_t> outputShape() const override { return {M_, N_}; }

  ThreadSize dispatchSize() const override {
    const auto &info = kMatMulVariants[resolvedVariant_];
    uint32_t gridX = ((N_ + info.effTileN - 1) / info.effTileN) * info.wgX;
    uint32_t gridY = ((M_ + info.effTileM - 1) / info.effTileM) * info.wgY;
    return {gridX, gridY, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    uint32_t strideA = (K_ + 3) & ~3u;
    uint32_t strideB = (N_ + 3) & ~3u;
    struct PushConstants {
      uint32_t M, K, N, strideA, strideB;
    } pc{M_, K_, N_, strideA, strideB};
    return toBytes(pc);
  }

private:
  std::vector<uint32_t> shapeA_;
  std::vector<uint32_t> shapeB_;
  DataType dtype_;
  uint32_t M_, K_, N_;
  int resolvedVariant_;
};

} // namespace cut
