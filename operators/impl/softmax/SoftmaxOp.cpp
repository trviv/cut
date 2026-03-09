#include "SoftmaxOp.h"
#include "SoftmaxVariants.generated.h"
#include "TensorStore.h"

namespace cut {

SoftmaxOpNode::SoftmaxOpNode(OperatorEnum op,
                             TensorStore &store,
                             const Tensor &a,
                             int dim,
                             std::optional<uint32_t> spec)
    : OpNode(op, store, spec) {
  const auto &buf = store.getTensor(a);
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

  // Output shape is the same as input shape
  outShape_ = shape;

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

  inputs_ = {a};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType SoftmaxOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> SoftmaxOpNode::shader() const {
  // Variant 0 = Softmax, Variant 1 = LogSoftmax
  int variantIndex = (op_ == OperatorEnum::LogSoftmax) ? 1 : 0;
  return getCompiledSoftmax(variantIndex, dtype_, dtype_);
}

std::vector<uint32_t> SoftmaxOpNode::outputShape() const {
  return outShape_;
}

ThreadSize SoftmaxOpNode::dispatchSize() const {
  uint32_t numSlices = outerSize_ * innerSize_;
  uint32_t gridX = numSlices * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> SoftmaxOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t outerSize;
    uint32_t reduceSize;
    uint32_t innerSize;
    uint32_t inOuterStride;
    uint32_t inReduceStride;
  } pc{outerSize_, reduceSize_, innerSize_, inOuterStride_, inReduceStride_};
  return toBytes(pc);
}

size_t SoftmaxOpNode::shaderKey() const {
  return shaderKeyWith(5);
}

} // namespace cut
