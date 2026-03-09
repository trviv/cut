#include "RepetitionPenaltyOp.h"
#include "SamplingShaders.generated.h"
#include "TensorStore.h"

namespace cut {

RepetitionPenaltyOpNode::RepetitionPenaltyOpNode(TensorStore &store,
                                                 const Tensor &logits,
                                                 const Tensor &penaltyFactors,
                                                 std::optional<uint32_t> spec)
    : OpNode(RepetitionPenalty, store, spec) {
  const auto &logitsBuf = store.getTensor(logits);
  const auto &factorsBuf = store.getTensor(penaltyFactors);
  outShape_ = logitsBuf.getShape();
  auto factorsShape = factorsBuf.getShape();
  if (outShape_ != factorsShape) {
    throw std::runtime_error(
        "RepetitionPenalty: logits and penaltyFactors must have same shape");
  }
  numElements_ = 1;
  for (auto d : outShape_)
    numElements_ *= d;
  inputs_ = {logits, penaltyFactors};
  output_ = store.createTensorEmpty(outputShape(), DataType::Float32);
}

DataType RepetitionPenaltyOpNode::outputDtype() const {
  return DataType::Float32;
}

std::optional<std::vector<uint32_t>> RepetitionPenaltyOpNode::shader() const {
  return compiledRepetitionPenalty(DataType::Float32, DataType::Float32);
}

std::vector<uint32_t> RepetitionPenaltyOpNode::outputShape() const {
  return outShape_;
}

ThreadSize RepetitionPenaltyOpNode::dispatchSize() const {
  // vec4 processing with 256 threads per workgroup
  uint32_t vec4Count = (numElements_ + 3) / 4;
  uint32_t gridX = ((vec4Count + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> RepetitionPenaltyOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numElements;
  } pc{numElements_};
  return toBytes(pc);
}

std::string RepetitionPenaltyOpNode::displayName() const {
  return "RepetitionPenalty";
}

} // namespace cut
