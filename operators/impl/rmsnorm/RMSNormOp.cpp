#include "RMSNormOp.h"
#include "RMSNormVariants.generated.h"
#include "TensorStore.h"

namespace cut {

RMSNormOpNode::RMSNormOpNode(TensorStore &store,
                             const Tensor &x,
                             const Tensor &weight,
                             float eps,
                             std::optional<uint32_t> spec)
    : OpNode(RMSNorm, store, spec), eps_(eps) {
  const auto &bufX = store.getTensor(x);
  const auto &bufWeight = store.getTensor(weight);

  const auto shapeX = bufX.getShape();
  const auto shapeWeight = bufWeight.getShape();

  dtype_ = bufX.getDtype();

  if (shapeX.size() != 1 || shapeWeight.size() != 1) {
    throw std::runtime_error("RMSNorm requires 1D tensors");
  }

  dim_ = shapeX[0];

  if (shapeWeight[0] != dim_) {
    throw std::runtime_error("RMSNorm: weight dimension must match input");
  }

  if (bufWeight.getDtype() != dtype_) {
    throw std::runtime_error("RMSNorm: weight must have same dtype as input");
  }

  // Align innermost dimension to multiple of 4
  alignedDim_ = (dim_ + 3) & ~3u;

  inputs_ = {x, weight};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType RMSNormOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> RMSNormOpNode::shader() const {
  // Use RMSNorm variant (index 0) from generated dispatch table
  return getCompiledRMSNorm(0, dtype_, dtype_);
}

std::vector<uint32_t> RMSNormOpNode::outputShape() const {
  return {dim_};
}

ThreadSize RMSNormOpNode::dispatchSize() const {
  // Single workgroup with 256 threads (for shared memory reduction)
  return {256, 1, 1};
}

std::vector<uint8_t> RMSNormOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t dim;
    uint32_t alignedDim;
    float eps;
  } pc{dim_, alignedDim_, eps_};
  return toBytes(pc);
}

std::string RMSNormOpNode::displayName() const {
  return "RMSNorm";
}

} // namespace cut
