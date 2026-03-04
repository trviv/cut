#include "ExtendedRMSNormOp.h"
#include "RMSNormVariants.generated.h"
#include "TensorStore.h"

namespace cut {

ExtendedRMSNormOpNode::ExtendedRMSNormOpNode(TensorStore &store,
                                             const Tensor &residual_base,
                                             const Tensor &delta,
                                             const Tensor &weight,
                                             float eps,
                                             std::optional<uint32_t> spec)
    : OpNode(ExtendedRMSNorm, store, spec), eps_(eps) {
  const auto &bufBase = store.getTensor(residual_base);
  const auto &bufDelta = store.getTensor(delta);
  const auto &bufWeight = store.getTensor(weight);

  const auto shapeBase = bufBase.getShape();
  const auto shapeDelta = bufDelta.getShape();
  const auto shapeWeight = bufWeight.getShape();

  dtype_ = bufBase.getDtype();

  if (shapeBase.size() != 1 || shapeDelta.size() != 1 ||
      shapeWeight.size() != 1) {
    throw std::runtime_error("ExtendedRMSNorm requires 1D tensors");
  }

  dim_ = shapeBase[0];

  if (shapeDelta[0] != dim_ || shapeWeight[0] != dim_) {
    throw std::runtime_error(
        "ExtendedRMSNorm: all inputs must have same dimension");
  }

  if (bufDelta.getDtype() != dtype_ || bufWeight.getDtype() != dtype_) {
    throw std::runtime_error(
        "ExtendedRMSNorm: all inputs must have same dtype");
  }

  // Align innermost dimension to multiple of 4
  alignedDim_ = (dim_ + 3) & ~3u;

  inputs_ = {residual_base, delta, weight};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType ExtendedRMSNormOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> ExtendedRMSNormOpNode::shader() const {
  // Use ExtendedRMSNorm variant (index 1) from generated dispatch table
  return getCompiledRMSNorm(1, dtype_, dtype_);
}

std::vector<uint32_t> ExtendedRMSNormOpNode::outputShape() const {
  return {dim_};
}

ThreadSize ExtendedRMSNormOpNode::dispatchSize() const {
  // Single workgroup with 256 threads (for shared memory reduction)
  return {256, 1, 1};
}

std::vector<uint8_t> ExtendedRMSNormOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t dim;
    uint32_t alignedDim;
    float eps;
  } pc{dim_, alignedDim_, eps_};
  return toBytes(pc);
}

std::string ExtendedRMSNormOpNode::displayName() const {
  return "ExtendedRMSNorm";
}

} // namespace cut
