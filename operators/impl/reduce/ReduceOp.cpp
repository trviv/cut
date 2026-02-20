#include "ReduceOp.h"

namespace cut {

namespace {
constexpr uint32_t kMultiReduceThreshold = 65536;

inline bool isMultiReduceCapable(OperatorEnum op) {
  switch (op) {
  case ReduceSum:
  case ReduceMean:
  case ReduceMin:
  case ReduceMax:
  case ReduceProd:
  case ReduceAny:
  case ReduceAll:
    return true;
  default:
    return false;
  }
}
} // namespace

// --- GlobalReduceOpNode ---

GlobalReduceOpNode::GlobalReduceOpNode(OperatorEnum op,
                                       std::vector<uint32_t> shape,
                                       DataType dtype,
                                       uint32_t innerDimSize)
    : op_(op), shape_(std::move(shape)), dtype_(dtype),
      numElements_(actualElementCount(shape_)), actualInner_(innerDimSize),
      alignedInner_((innerDimSize + 3) & ~static_cast<uint32_t>(3)) {}

OperatorEnum GlobalReduceOpNode::op() const {
  return op_;
}
DataType GlobalReduceOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> GlobalReduceOpNode::outputShape() const {
  return {1};
}

bool GlobalReduceOpNode::isMultiPass() const {
  return numElements_ > kMultiReduceThreshold && isMultiReduceCapable(op_);
}

size_t GlobalReduceOpNode::executionSize() const {
  return numElements_;
}

ThreadSize GlobalReduceOpNode::dispatchSize() const {
  return {256, 1, 1};
}

std::vector<uint8_t> GlobalReduceOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numElements;
    uint32_t actualInner;
    uint32_t alignedInner;
  } pc{static_cast<uint32_t>(numElements_), actualInner_, alignedInner_};
  return toBytes(pc);
}

// --- DimReduceOpNode ---

DimReduceOpNode::DimReduceOpNode(OperatorEnum op,
                                 std::vector<uint32_t> shape,
                                 int dim,
                                 DataType dtype,
                                 uint32_t bufInnerDimSize)
    : op_(op), shape_(std::move(shape)), dtype_(dtype),
      bufInnerDim_(bufInnerDimSize),
      alignedBufInner_((bufInnerDimSize + 3) & ~static_cast<uint32_t>(3)) {
  int ndim = static_cast<int>(shape_.size());
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
    outerSize_ *= shape_[i];
  reduceSize_ = shape_[dim_];
  innerSize_ = 1;
  for (int i = dim_ + 1; i < ndim; ++i)
    innerSize_ *= shape_[i];

  for (int i = 0; i < ndim; ++i) {
    if (i != dim_)
      outShape_.push_back(shape_[i]);
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
}

OperatorEnum DimReduceOpNode::op() const {
  return op_;
}
DataType DimReduceOpNode::shaderDtype() const {
  return dtype_;
}
bool DimReduceOpNode::isDimReduce() const {
  return true;
}
OperatorEnum DimReduceOpNode::baseReduceOp() const {
  return op_;
}

std::vector<uint32_t> DimReduceOpNode::outputShape() const {
  return outShape_;
}

ThreadSize DimReduceOpNode::dispatchSize() const {
  uint32_t numOutputs = outerSize_ * innerSize_;
  uint32_t gridX = ((numOutputs + 255) / 256) * 256;
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

// --- NormOpNode ---

NormOpNode::NormOpNode(std::vector<uint32_t> shape, DataType dtype)
    : shape_(std::move(shape)), dtype_(dtype),
      numElements_(actualElementCount(shape_)) {}

OperatorEnum NormOpNode::op() const {
  return Norm;
}
DataType NormOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> NormOpNode::outputShape() const {
  return {1};
}

ThreadSize NormOpNode::dispatchSize() const {
  return {256, 1, 1};
}

std::vector<uint8_t> NormOpNode::pushConstants() const {
  uint32_t n = static_cast<uint32_t>(numElements_);
  return toBytes(n);
}

size_t NormOpNode::executionSize() const {
  return numElements_;
}

// --- DotOpNode ---

DotOpNode::DotOpNode(std::vector<uint32_t> shapeA,
                     std::vector<uint32_t> shapeB,
                     DataType dtype)
    : shapeA_(std::move(shapeA)), shapeB_(std::move(shapeB)), dtype_(dtype) {
  if (actualElementCount(shapeA_) != actualElementCount(shapeB_)) {
    throw std::runtime_error(
        "Vector size mismatch: " + std::to_string(actualElementCount(shapeA_)) +
        " vs " + std::to_string(actualElementCount(shapeB_)));
  }
  count_ = static_cast<uint32_t>(actualElementCount(shapeA_));
  numWorkgroups_ = (count_ + 255) / 256;
}

OperatorEnum DotOpNode::op() const {
  return Dot;
}
DataType DotOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> DotOpNode::outputShape() const {
  return {numWorkgroups_};
}

ThreadSize DotOpNode::dispatchSize() const {
  return {numWorkgroups_ * 256, 1, 1};
}

std::vector<uint8_t> DotOpNode::pushConstants() const {
  return toBytes(count_);
}

std::vector<ComputeBinding> DotOpNode::handleBindings() const {
  // Dot: A at 0, B at 1, partials at 2
  std::vector<ComputeBinding> bindings;
  for (uint32_t i = 0; i < static_cast<uint32_t>(inputs_.size()); ++i) {
    bindings.emplace_back(i, inputs_[i]);
  }
  if (hasOutput_) {
    bindings.emplace_back(static_cast<uint32_t>(inputs_.size()), output_);
  }
  return bindings;
}

// --- CumOpNode ---

CumOpNode::CumOpNode(OperatorEnum op,
                     std::vector<uint32_t> shape,
                     int dim,
                     DataType dtype,
                     uint32_t bufInnerDimSize)
    : op_(op), shape_(std::move(shape)), dtype_(dtype),
      bufInnerDim_(bufInnerDimSize),
      alignedBufInner_((bufInnerDimSize + 3) & ~static_cast<uint32_t>(3)) {
  int ndim = static_cast<int>(shape_.size());
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
    outerSize_ *= shape_[i];
  reduceSize_ = shape_[dim_];
  innerSize_ = 1;
  for (int i = dim_ + 1; i < ndim; ++i)
    innerSize_ *= shape_[i];

  inReduceStride_ = innerSize_;
  inOuterStride_ = reduceSize_ * innerSize_;
  if (innerSize_ == bufInnerDim_) {
    inReduceStride_ = alignedBufInner_;
    inOuterStride_ = reduceSize_ * alignedBufInner_;
  } else if (innerSize_ == 1) {
    inReduceStride_ = 1;
    inOuterStride_ = alignedBufInner_;
  }
}

OperatorEnum CumOpNode::op() const {
  return op_;
}
DataType CumOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> CumOpNode::outputShape() const {
  return shape_;
}

ThreadSize CumOpNode::dispatchSize() const {
  uint32_t numOutputs = outerSize_ * innerSize_;
  uint32_t gridX = ((numOutputs + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> CumOpNode::pushConstants() const {
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
