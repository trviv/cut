#include "ReduceDimOp.h"
#include "Runtime.h"
#include "Shaders.h"

namespace cut {

DimReduceOpNode::DimReduceOpNode(OperatorEnum op,
                                 Runtime &runtime,
                                 const Tensor &a,
                                 int dim,
                                 std::optional<uint32_t> spec)
    : OpNode(op, runtime, spec) {
  const auto &buf = runtime.getTensor(a);
  const auto shape = buf.getShape();
  dtype_ = buf.getDtype();
  bufInnerDim_ = buf.innerDimSize();
  alignedBufInner_ = (bufInnerDim_ + 3) & ~static_cast<uint32_t>(3);
  int ndim = static_cast<int>(shape.size());
  if (dim < 0)
    dim = ndim + dim;
  dim_ = dim;
  if (dim_ < 0 || dim_ >= ndim) {
    throw std::invalid_argument("dim " + std::to_string(dim_) +
                                " out of range for tensor with " +
                                std::to_string(ndim) + " dimensions");
  }

  outerSize_ = 1;
  for (int i = 0; i < dim_; ++i)
    outerSize_ *= shape[i];
  reduceSize_ = shape[dim_];
  innerSize_ = 1;
  for (int i = dim_ + 1; i < ndim; ++i)
    innerSize_ *= shape[i];

  for (int i = 0; i < ndim; ++i) {
    if (i != dim_)
      outShape_.push_back(shape[i]);
  }
  if (outShape_.empty())
    outShape_.push_back(1);

  // Compute input buffer strides accounting for inner-dim alignment
  inReduceStride_ = innerSize_;
  inOuterStride_ = reduceSize_ * innerSize_;
  if (innerSize_ == bufInnerDim_) {
    inReduceStride_ = alignedBufInner_;
    inOuterStride_ = reduceSize_ * alignedBufInner_;
  } else if (innerSize_ == 1) {
    inReduceStride_ = 1;
    inOuterStride_ = alignedBufInner_;
  }
  resolvedVariant_ = spec.value_or(kReduceDimDefaultVariant);
  inputs_ = {a};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType DimReduceOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<uint32_t> DimReduceOpNode::spec() const {
  return resolvedVariant_;
}

const std::optional<std::vector<uint32_t>> &DimReduceOpNode::shader() const {
  if (!shader_.has_value()) {
    shader_ = getDimReduceShader(op_, dtype_, resolvedVariant_);
  }
  return shader_;
}

size_t DimReduceOpNode::shaderKey() const {
  return static_cast<size_t>(op_) | (static_cast<size_t>(dtype_) << 16) |
         (size_t(1) << 48) | (static_cast<size_t>(resolvedVariant_) << 32);
}

std::vector<uint32_t> DimReduceOpNode::outputShape() const {
  return outShape_;
}

ThreadSize DimReduceOpNode::dispatchSize() const {
  uint32_t numOutputs = outerSize_ * innerSize_;
  if (resolvedVariant_ == 0) {
    // Naive: one thread per output element
    uint32_t gridX = ((numOutputs + 255) / 256) * 256;
    return {gridX, 1, 1};
  }
  // Shared: one workgroup (256 threads) per output element
  uint32_t gridX = numOutputs * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> DimReduceOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t outerSize;
    uint32_t reduceSize;
    uint32_t innerSize;
    uint32_t inOuterStride;
    uint32_t inReduceStride;
  } pc{outerSize_, reduceSize_, innerSize_, inOuterStride_, inReduceStride_};
  return toBytes(pc);
}

} // namespace cut
