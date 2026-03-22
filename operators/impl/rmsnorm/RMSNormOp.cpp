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

  if (shapeX.size() < 1 || shapeX.size() > 2 || shapeWeight.size() != 1) {
    throw std::runtime_error("RMSNorm requires 1D or 2D input, 1D weight");
  }

  // 2D input [batchSize, dim]: normalize each row independently.
  // 1D input [dim]: batchSize=1 (backward compatible).
  if (shapeX.size() == 2) {
    batchSize_ = shapeX[0];
    dim_ = shapeX[1];
  } else {
    batchSize_ = 1;
    dim_ = shapeX[0];
  }

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
  if (batchSize_ > 1)
    return {batchSize_, dim_};
  return {dim_};
}

ThreadSize RMSNormOpNode::dispatchSize() const {
  // One workgroup of 256 threads per row. Y dimension = batchSize.
  return {256, batchSize_, 1};
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
