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

  cudaRowMapping_ = store.caps().backend == ComputeBackend::CUDA;

  const bool isLog = op_ == OperatorEnum::LogSoftmax;
  variant_ = isLog ? kSoftmaxVariantLogSoftmax : kSoftmaxVariantSoftmax;

  // A row that fits in a block's registers is read once; past that the kernel
  // streams and reads it twice. Doubling the block to 512 threads doubles the
  // rows that fit, which is worth a whole variant for the band in between:
  // measured on a 16384-column attention softmax it is 1.5x, exactly the
  // 3-pass/2-pass traffic ratio. Outside that band the wide variant is a
  // pessimisation, so it is not the default — a 512-thread block halves the
  // blocks resident per SM, and on short rows that costs more than the pass it
  // would save (measured 0.61x at 4096x128). Only the band gets it.
  if (reduceSize_ > softmaxRegisterResidentCols(256) &&
      reduceSize_ <= softmaxRegisterResidentCols(512)) {
    variant_ = isLog ? kSoftmaxVariantLogSoftmaxWide : kSoftmaxVariantSoftmaxWide;
  }
  if (spec.has_value() && *spec < static_cast<uint32_t>(kSoftmaxVariantCount))
    variant_ = *spec;
  // Fall back if this dtype has no shader for the chosen variant.
  if (!getCompiledSoftmax(static_cast<int>(variant_), dtype_, dtype_).has_value())
    variant_ = isLog ? kSoftmaxVariantLogSoftmax : kSoftmaxVariantSoftmax;

  wgSize_ = kSoftmaxVariants[variant_].wgX;
  // Carried in spec_ so it reaches shaderKey(), which keys the pipeline cache —
  // two variants of one op that hashed alike would share a compiled shader.
  spec_ = variant_;

  inputs_ = {a};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType SoftmaxOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> SoftmaxOpNode::shader() const {
  return getCompiledSoftmax(static_cast<int>(variant_), dtype_, dtype_);
}

std::vector<uint32_t> SoftmaxOpNode::outputShape() const {
  return outShape_;
}

ThreadSize SoftmaxOpNode::dispatchSize() const {
  uint32_t numSlices = outerSize_ * innerSize_;
  if (!cudaRowMapping_) {
    // HLSL/SPIR-V shape: one workgroup per slice, always.
    return {numSlices * wgSize_, 1, 1};
  }
  // The native CUDA kernel gives a short row one warp rather than a whole
  // block, so a block covers wgSize/threadsPerRow rows and the grid shrinks by
  // the same factor. SoftmaxCommon.cuh recomputes this split from reduceSize.
  uint32_t rowsPerBlock = wgSize_ / softmaxThreadsPerRow(reduceSize_, wgSize_);
  uint32_t blocks = (numSlices + rowsPerBlock - 1) / rowsPerBlock;
  return {blocks * wgSize_, 1, 1};
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
