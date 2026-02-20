#pragma once

#include "OpNode.h"
#include "impl/matmul/MatMulVariants.generated.h"

namespace cut {

class MatMulOpNode : public OpNode {
public:
  MatMulOpNode(std::vector<uint32_t> shapeA,
               std::vector<uint32_t> shapeB,
               DataType dtype,
               int variantIdx = -1);

  DataType shaderDtype() const override;
  DataType outputDtype() const override;
  int variantIndex() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  std::vector<uint32_t> shapeA_;
  std::vector<uint32_t> shapeB_;
  DataType dtype_;
  uint32_t M_, K_, N_;
  int resolvedVariant_;
};

} // namespace cut
