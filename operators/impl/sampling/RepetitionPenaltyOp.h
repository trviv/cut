#pragma once

#include "OpNode.h"

namespace cut {

/// GPU repetition penalty: applies conditional scaling to logits.
/// For each element: logit > 0 ? logit/factor : logit*factor.
/// factor == 1.0 for non-penalized tokens (identity).
class RepetitionPenaltyOpNode : public OpNode {
public:
  RepetitionPenaltyOpNode(TensorStore &store,
                          const Tensor &logits,
                          const Tensor &penaltyFactors,
                          std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::string displayName() const override;

private:
  uint32_t numElements_;
  std::vector<uint32_t> outShape_;
};

} // namespace cut
