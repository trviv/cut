#include "ReduceOp.h"
#include "Dispatcher.h"
#include "ReduceShaders.generated.h"
#include "Runtime.h"
#include "Shaders.h"

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
                                       Runtime &runtime,
                                       const Tensor &a,
                                       std::optional<uint32_t> spec)
    : OpNode(op, runtime, spec) {
  const auto &buf = runtime.getTensor(a);
  dtype_ = buf.getDtype();
  numElements_ = actualElementCount(buf.getShape());
  actualInner_ = buf.innerDimSize();
  alignedInner_ = (actualInner_ + 3) & ~static_cast<uint32_t>(3);
  inputs_ = {a};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
}

DataType GlobalReduceOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> GlobalReduceOpNode::shader() const {
  std::optional<std::vector<uint32_t>> compiled;
  if (op_ == ReduceArgmax || op_ == ReduceArgmin) {
    compiled = compiledReduceArg(dtype_);
  } else if (op_ == Norm) {
    compiled = compiledNorm(dtype_);
  } else {
    compiled = compiledReduce(dtype_);
  }
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
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
  if (op_ == ReduceArgmax || op_ == ReduceArgmin || op_ == Norm) {
    uint32_t n = static_cast<uint32_t>(numElements_);
    return toBytes(n);
  }
  struct PushConstants {
    uint32_t numElements;
    uint32_t actualInner;
    uint32_t alignedInner;
  } pc{static_cast<uint32_t>(numElements_), actualInner_, alignedInner_};
  return toBytes(pc);
}

void GlobalReduceOpNode::buildSubOperations(Dispatcher &dispatcher) {
  uint32_t numElements = static_cast<uint32_t>(numElements_);

  // Each WG of 256 threads processes ~1024 elements, cap at 256 workgroups
  uint32_t groupCount = (numElements + 1023) / 1024;
  groupCount = std::min(groupCount, 256u);
  groupCount = std::max(groupCount, 2u);

  Tensor inputHandle = inputs_[0];
  Tensor outputHandle = output_;

  Tensor partialSums = dispatcher.acquireTempBuffer(groupCount, dtype_);

  // Phase 1: Partial reduce — each workgroup reduces its batch
  struct PartialPC {
    uint32_t numElements;
    uint32_t groupCount;
    uint32_t reduceOp;
  } partialPC{numElements, groupCount, static_cast<uint32_t>(op_)};
  subOps_.push_back(std::make_unique<InternalOpNode>(
      InternalPartialReduce, dtype_,
      std::vector<Tensor>{inputHandle, partialSums},
      ThreadSize{256 * groupCount, 1, 1}, toBytes(partialPC), true));

  // Phase 2: Final reduce — single workgroup reduces partial sums
  struct FinalPC {
    uint32_t numElements;
    uint32_t originalNumElements;
    uint32_t reduceOp;
  } finalPC{groupCount, numElements, static_cast<uint32_t>(op_)};
  subOps_.push_back(std::make_unique<InternalOpNode>(
      InternalFinalReduce, dtype_,
      std::vector<Tensor>{partialSums, outputHandle}, ThreadSize{256, 1, 1},
      toBytes(finalPC)));
}

// --- NormOpNode ---

NormOpNode::NormOpNode(Runtime &runtime,
                       const Tensor &a,
                       std::optional<uint32_t> spec)
    : OpNode(Norm, runtime, spec) {
  const auto &buf = runtime.getTensor(a);
  dtype_ = buf.getDtype();
  numElements_ = actualElementCount(buf.getShape());
  inputs_ = {a};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
}

DataType NormOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> NormOpNode::shader() const {
  auto compiled = compiledNorm(dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
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

DotOpNode::DotOpNode(Runtime &runtime,
                     const Tensor &a,
                     const Tensor &b,
                     std::optional<uint32_t> spec)
    : OpNode(Dot, runtime, spec) {
  const auto &bufA = runtime.getTensor(a);
  const auto &bufB = runtime.getTensor(b);
  const auto shapeA = bufA.getShape();
  const auto shapeB = bufB.getShape();
  dtype_ = bufA.getDtype();
  if (actualElementCount(shapeA) != actualElementCount(shapeB)) {
    throw std::runtime_error(
        "Vector size mismatch: " + std::to_string(actualElementCount(shapeA)) +
        " vs " + std::to_string(actualElementCount(shapeB)));
  }
  count_ = static_cast<uint32_t>(actualElementCount(shapeA));
  numWorkgroups_ = (count_ + 255) / 256;
  inputs_ = {a, b};
  output_ = runtime.createTensorEmpty(outputShape(), DataType::Float32);
}

DataType DotOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> DotOpNode::shader() const {
  auto compiled = compiledDot(dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
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

// --- CumOpNode ---

CumOpNode::CumOpNode(OperatorEnum op,
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
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
}

DataType CumOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> CumOpNode::shader() const {
  auto compiled = compiledCumOp(dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> CumOpNode::outputShape() const {
  return runtime_->getTensor(inputs_[0]).getShape();
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
