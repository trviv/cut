#pragma once

#include "OpNode.h"
#include "impl/transpose/TransposeVariants.generated.h"

namespace cut {

class TransposeOpNode : public OpNode {
public:
  TransposeOpNode(TensorStore &store,
                  const Tensor &a,
                  std::optional<uint32_t> spec = {});

  LogicalOpType logicalType() const override {
    return LogicalOpType::Transpose;
  }

  DataType shaderDtype() const override;
  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t M_, N_;
};

} // namespace cut
