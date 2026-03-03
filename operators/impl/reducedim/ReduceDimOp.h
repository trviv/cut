#pragma once

#include "OpNode.h"
#include "impl/reducedim/ReduceDimVariants.generated.h"

namespace cut {

class DimReduceOpNode : public OpNode {
public:
  DimReduceOpNode(OperatorEnum op,
                  TensorStore &store,
                  const Tensor &a,
                  int dim,
                  std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  int dim_;
  uint32_t outerSize_, reduceSize_, innerSize_;
  uint32_t inReduceStride_, inOuterStride_;
  uint32_t bufInnerDim_, alignedBufInner_;
  std::vector<uint32_t> outShape_;
};

} // namespace cut
