#pragma once

#include "OpNode.h"

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

class GlobalReduceOpNode : public OpNode {
public:
  GlobalReduceOpNode(OperatorEnum op,
                     std::vector<uint32_t> shape,
                     DataType dtype,
                     uint32_t innerDimSize)
      : op_(op), shape_(std::move(shape)), dtype_(dtype),
        numElements_(actualElementCount(shape_)), actualInner_(innerDimSize),
        alignedInner_((innerDimSize + 3) & ~static_cast<uint32_t>(3)) {}

  void validate() const override {}

  OperatorEnum op() const override { return op_; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return {1}; }

  bool isMultiPass() const override {
    return numElements_ > kMultiReduceThreshold && isMultiReduceCapable(op_);
  }

  size_t executionSize() const override { return numElements_; }

  ThreadSize dispatchSize() const override { return {256, 1, 1}; }

  std::vector<uint8_t> pushConstants() const override {
    struct PushConstants {
      uint32_t numElements;
      uint32_t actualInner;
      uint32_t alignedInner;
    } pc{static_cast<uint32_t>(numElements_), actualInner_, alignedInner_};
    return toBytes(pc);
  }

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  size_t numElements_;
  uint32_t actualInner_;
  uint32_t alignedInner_;
};

class DimReduceOpNode : public OpNode {
public:
  DimReduceOpNode(OperatorEnum op,
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

  void validate() const override {
    int ndim = static_cast<int>(shape_.size());
    if (dim_ < 0 || dim_ >= ndim) {
      throw std::invalid_argument("dim " + std::to_string(dim_) +
                                  " out of range for tensor with " +
                                  std::to_string(ndim) + " dimensions");
    }
  }

  OperatorEnum op() const override { return op_; }
  DataType shaderDtype() const override { return dtype_; }
  bool isDimReduce() const override { return true; }
  OperatorEnum baseReduceOp() const override { return op_; }

  std::vector<uint32_t> outputShape() const override { return outShape_; }

  ThreadSize dispatchSize() const override {
    uint32_t numOutputs = outerSize_ * innerSize_;
    uint32_t gridX = ((numOutputs + 255) / 256) * 256;
    return {gridX, 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    struct PushConstants {
      uint32_t outerSize;
      uint32_t reduceSize;
      uint32_t innerSize;
      uint32_t inOuterStride;
      uint32_t inReduceStride;
    } pc{outerSize_, reduceSize_, innerSize_, inOuterStride_, inReduceStride_};
    return toBytes(pc);
  }

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  int dim_;
  uint32_t outerSize_, reduceSize_, innerSize_;
  uint32_t inReduceStride_, inOuterStride_;
  uint32_t bufInnerDim_, alignedBufInner_;
  std::vector<uint32_t> outShape_;
};

class NormOpNode : public OpNode {
public:
  NormOpNode(std::vector<uint32_t> shape, DataType dtype)
      : shape_(std::move(shape)), dtype_(dtype),
        numElements_(actualElementCount(shape_)) {}

  void validate() const override {}

  OperatorEnum op() const override { return Norm; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return {1}; }

  ThreadSize dispatchSize() const override { return {256, 1, 1}; }

  std::vector<uint8_t> pushConstants() const override {
    uint32_t n = static_cast<uint32_t>(numElements_);
    return toBytes(n);
  }

  size_t executionSize() const override { return numElements_; }

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  size_t numElements_;
};

class DotOpNode : public OpNode {
public:
  DotOpNode(std::vector<uint32_t> shapeA,
            std::vector<uint32_t> shapeB,
            DataType dtype)
      : shapeA_(std::move(shapeA)), shapeB_(std::move(shapeB)), dtype_(dtype) {
    count_ = static_cast<uint32_t>(actualElementCount(shapeA_));
    numWorkgroups_ = (count_ + 255) / 256;
  }

  void validate() const override {
    if (actualElementCount(shapeA_) != actualElementCount(shapeB_)) {
      throw std::runtime_error("Vector size mismatch: " +
                               std::to_string(actualElementCount(shapeA_)) +
                               " vs " +
                               std::to_string(actualElementCount(shapeB_)));
    }
  }

  OperatorEnum op() const override { return Dot; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override {
    return {numWorkgroups_};
  }

  ThreadSize dispatchSize() const override {
    return {numWorkgroups_ * 256, 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    return toBytes(count_);
  }

  std::vector<ComputeBinding> handleBindings() const override {
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

private:
  std::vector<uint32_t> shapeA_;
  std::vector<uint32_t> shapeB_;
  DataType dtype_;
  uint32_t count_;
  uint32_t numWorkgroups_;
};

class CumOpNode : public OpNode {
public:
  CumOpNode(OperatorEnum op,
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

  void validate() const override {
    int ndim = static_cast<int>(shape_.size());
    if (dim_ < 0 || dim_ >= ndim) {
      throw std::invalid_argument("dim " + std::to_string(dim_) +
                                  " out of range for tensor with " +
                                  std::to_string(ndim) + " dimensions");
    }
  }

  OperatorEnum op() const override { return op_; }
  DataType shaderDtype() const override { return dtype_; }

  // CumOp output is same shape as input (not reduced)
  std::vector<uint32_t> outputShape() const override { return shape_; }

  ThreadSize dispatchSize() const override {
    uint32_t numOutputs = outerSize_ * innerSize_;
    uint32_t gridX = ((numOutputs + 255) / 256) * 256;
    return {gridX, 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    struct PushConstants {
      uint32_t outerSize;
      uint32_t reduceSize;
      uint32_t innerSize;
      uint32_t inOuterStride;
      uint32_t inReduceStride;
    } pc{outerSize_, reduceSize_, innerSize_, inOuterStride_, inReduceStride_};
    return toBytes(pc);
  }

private:
  OperatorEnum op_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  int dim_;
  uint32_t outerSize_, reduceSize_, innerSize_;
  uint32_t inReduceStride_, inOuterStride_;
  uint32_t bufInnerDim_, alignedBufInner_;
};

} // namespace cut
