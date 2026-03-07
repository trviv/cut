#include "Operations.h"
#include "OpNode.h"
#include "Runtime.h"
#include "ShapeUtils.h"
#include "graph/GraphExecutor.h"

#include "impl/avgpool2d/AvgPool2DOp.h"
#include "impl/binary/BinaryOp.h"
#include "impl/cast/CastOp.h"
#include "impl/conv1d/Conv1DOp.h"
#include "impl/conv2d/Conv2DOp.h"
#include "impl/dequant/DequantOp.h"
#include "impl/matmul/MatMulBinaryOp.h"
#include "impl/matmul/MatMulOp.h"
#include "impl/matmul/MatMulSiLUOp.h"
#include "impl/maxpool2d/MaxPool2DOp.h"
#include "impl/memory/MemoryOp.h"
#include "impl/reduce/ReduceOp.h"
#include "impl/reducedim/ReduceDimOp.h"
#include "impl/rmsnorm/ExtendedRMSNormOp.h"
#include "impl/rmsnorm/RMSNormOp.h"
#include "impl/scan/ScanOp.h"
#include "impl/sort/SortOp.h"
#include "impl/ternary/TernaryOp.h"
#include "impl/transpose/TransposeOp.h"
#include "impl/unary/UnaryOp.h"

#include "impl/attention/AttentionOp.h"
#include "impl/rope/RoPEOp.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cut {

using graph::Graph;

// =========================================================================
// Operations
// =========================================================================

Operations::Operations(Runtime &runtime)
    : runtime_(&runtime), store_(&runtime.store()),
      graph_(std::make_unique<Graph>()) {}

std::vector<uint32_t> Operations::getShape(const Tensor &h) const {
  return store_->getTensor(h).getShape();
}

DataType Operations::getDtype(const Tensor &h) const {
  return store_->getTensor(h).getDtype();
}

// =========================================================================
// Graph mode helpers
// =========================================================================

void Operations::flush() {
  if (graph_->size() == 0 || graph_->isExecuted())
    return;

  // Mark all non-input nodes as graph outputs
  for (uint32_t i = 0; i < graph_->size(); ++i) {
    auto &n = graph_->nodes()[i];
    if (n.op && !n.isRemoved && !n.isInput) {
      graph_->markOutput(i);
    }
  }

  // Mark as executed before running the executor,
  // so that recursive flush() calls (from dispatch) are no-ops.
  graph_->markExecuted();

  // Execute without optimizing.  Auto-flush marks every non-input node as
  // an output, and the user holds placeholder tensors for those nodes.
  // Optimizer passes (e.g. IdentityReshapePass) may replace output nodes,
  // causing the executor to write results into the replacement node's buffer
  // rather than the placeholder the user holds.  Optimization is applied by
  // GraphBuilder::build() where outputs are explicitly marked.
  graph::GraphExecutor executor(*this, *store_);
  executor.execute(*graph_);

  // Replace with a fresh graph to release OpNode buffer references.
  graph_ = std::make_unique<graph::Graph>();
  castCache_.clear();
}

std::unique_ptr<graph::Graph> Operations::takeGraph() {
  auto result = std::move(graph_);
  graph_ = std::make_unique<graph::Graph>();
  castCache_.clear();
  return result;
}

void Operations::markGraphOutput(const Tensor &t) {
  graph_->markOutput(t);
}

void Operations::ensureFreshGraph() {
  if (graph_->isExecuted()) {
    graph_ = std::make_unique<graph::Graph>();
    castCache_.clear();
  }
}

uint32_t Operations::toNodeId(const Tensor &t) {
  auto id = graph_->tryNodeId(t);
  if (id)
    return *id;
  // Register as a new graph input
  registerInput(t, false);
  return graph_->nodeId(t);
}

Tensor Operations::registerInput(const Tensor &gpuHandle, bool isConstant) {
  ensureFreshGraph();
  // Deduplicate only constant inputs (weights). Dynamic inputs may reuse the
  // same placeholder handle for distinct graph inputs that receive different
  // tensors at execution time, so each call must create a separate node.
  if (isConstant) {
    if (graph_->tryNodeId(gpuHandle))
      return gpuHandle;
  }
  auto shape = getShape(gpuHandle);
  auto dtype = getDtype(gpuHandle);
  auto node =
      std::make_unique<InputOpNode>(gpuHandle, shape, dtype, isConstant);
  graph_->addNode(std::move(node), gpuHandle);
  return gpuHandle;
}

Tensor Operations::recordOrEncode(std::unique_ptr<OpNode> node) {
  ensureFreshGraph();
  Tensor output = node->output();
  std::vector<uint32_t> inputIds;
  for (const auto &inp : node->inputs()) {
    inputIds.push_back(toNodeId(inp));
  }
  graph_->addNode(std::move(node), output, std::move(inputIds));
  return output;
}

void Operations::setProfilingEnabled(bool enabled) {
  runtime_->setProfilingEnabled(enabled);
}

void Operations::dispatch(std::unique_ptr<OpNode> node) {
  flush();
  runtime_->dispatch(std::move(node));
}

void Operations::dispatch(OpNode &node) {
  flush();
  runtime_->dispatch(node);
}

// =========================================================================
// Generic element-wise ops
// =========================================================================

Tensor Operations::binaryOp(OperatorEnum op,
                            const Tensor &a,
                            const TensorLike &b,
                            std::optional<uint32_t> spec) {
  auto node = std::make_unique<BinaryOpNode>(op, *store_, a, b, spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::unaryOp(OperatorEnum op,
                           const Tensor &a,
                           std::optional<uint32_t> spec) {
  auto node = std::make_unique<UnaryOpNode>(op, *store_, a, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Reduction ops
// =========================================================================

Tensor Operations::reduce(OperatorEnum op,
                          const Tensor &a,
                          std::optional<int> dim,
                          std::optional<uint32_t> spec) {
  if (!dim.has_value()) {
    auto node = std::make_unique<GlobalReduceOpNode>(op, *store_, a, spec);
    return recordOrEncode(std::move(node));
  }
  auto node =
      std::make_unique<DimReduceOpNode>(op, *store_, a, dim.value(), spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Matrix ops
// =========================================================================

Tensor Operations::matmul(const Tensor &a,
                          const Tensor &b,
                          std::optional<uint32_t> spec) {
  auto node = std::make_unique<MatMulOpNode>(*store_, a, b, spec);
  auto inputs = resolveAndCastInputs(*node, {a, b});
  if (inputs[0] != a || inputs[1] != b)
    node = std::make_unique<MatMulOpNode>(*store_, inputs[0], inputs[1], spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::matmul(const Tensor &a,
                          const Tensor &packedB,
                          const Tensor &scales,
                          std::optional<uint32_t> spec) {
  auto node = std::make_unique<MatMulOpNode>(*store_, a, packedB, scales, spec);
  auto inputs = resolveAndCastInputs(*node, {a, packedB, scales});
  if (inputs[0] != a)
    node = std::make_unique<MatMulOpNode>(*store_, inputs[0], packedB, scales,
                                          spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::matmulSiLU(const Tensor &a,
                              const Tensor &b,
                              std::optional<uint32_t> spec) {
  auto node = std::make_unique<MatMulSiLUOpNode>(*store_, a, b, spec);
  auto inputs = resolveAndCastInputs(*node, {a, b});
  if (inputs[0] != a || inputs[1] != b)
    node =
        std::make_unique<MatMulSiLUOpNode>(*store_, inputs[0], inputs[1], spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::matmulSiLU(const Tensor &a,
                              const Tensor &packedB,
                              const Tensor &scales,
                              std::optional<uint32_t> spec) {
  auto node =
      std::make_unique<MatMulSiLUOpNode>(*store_, a, packedB, scales, spec);
  auto inputs = resolveAndCastInputs(*node, {a, packedB, scales});
  if (inputs[0] != a)
    node = std::make_unique<MatMulSiLUOpNode>(*store_, inputs[0], packedB,
                                              scales, spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::matmulBinary(OperatorEnum binaryOp,
                                const Tensor &a,
                                const Tensor &packedB,
                                const Tensor &scales,
                                const Tensor &d,
                                std::optional<uint32_t> spec) {
  auto node = std::make_unique<MatMulBinaryOpNode>(*store_, binaryOp, a,
                                                   packedB, scales, d, spec);
  auto inputs = resolveAndCastInputs(*node, {a, packedB, scales, d});
  if (inputs[0] != a || inputs[3] != d)
    node = std::make_unique<MatMulBinaryOpNode>(
        *store_, binaryOp, inputs[0], packedB, scales, inputs[3], spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::transpose(const Tensor &a, std::optional<uint32_t> spec) {
  auto node = std::make_unique<TransposeOpNode>(*store_, a, spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::dot(const Tensor &a,
                       const Tensor &b,
                       std::optional<uint32_t> spec) {
  return reduce(ReduceSum, binaryOp(BinaryMul, a, b));
}

// =========================================================================
// Special ops
// =========================================================================

Tensor Operations::clamp(const Tensor &a,
                         DataReference clampData,
                         std::optional<uint32_t> spec) {
  uint32_t minBits = 0, maxBits = 0;
  std::memcpy(&minBits, clampData.ptr, sizeof(uint32_t));
  std::memcpy(&maxBits,
              static_cast<const uint8_t *>(clampData.ptr) + sizeof(uint32_t),
              sizeof(uint32_t));
  auto node =
      std::make_unique<TernaryClampOpNode>(*store_, a, minBits, maxBits, spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::where(const Tensor &cond,
                         const Tensor &x,
                         const Tensor &y,
                         std::optional<uint32_t> spec) {
  auto node = std::make_unique<TernarySelectOpNode>(*store_, cond, x, y, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Cumulative ops
// =========================================================================

Tensor Operations::cumOp(const Tensor &a,
                         OperatorEnum op,
                         std::optional<int> dim,
                         std::optional<uint32_t> spec) {
  int d = dim.value_or(0);
  auto node = std::make_unique<CumOpNode>(op, *store_, a, d, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Statistical ops
// =========================================================================

Tensor
Operations::variance(const Tensor &a, int correction, std::optional<int> dim) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);

  if (!dim.has_value()) {
    // Global variance: var = sum((x - mean)^2) / (n - correction)
    // All ops on GPU via reduce + expand + elementwise.
    size_t n = shapeProduct(shape);
    Tensor flat = reshape(a, {static_cast<int32_t>(n)});
    Tensor meanVal = reduce(OperatorEnum::ReduceMean, flat);
    Tensor meanExp = expand(meanVal, {static_cast<uint32_t>(n)});
    Tensor diff = binaryOp(OperatorEnum::BinarySub, flat, meanExp);
    Tensor diffSq = unaryOp(OperatorEnum::UnarySquare, diff);
    Tensor sumSq = reduce(OperatorEnum::ReduceSum, diffSq);
    int denom = static_cast<int>(n) - correction;
    if (denom > 0) {
      return binaryOp(OperatorEnum::BinaryDiv, sumSq,
                      static_cast<float>(denom));
    }
    return full({1}, 0.0f, dtype);
  }

  // Dimension-wise variance: var(x, dim) = mean((x - mean(x, dim))^2, dim)
  // adjusted for Bessel's correction.
  // All ops on GPU via reduce + unsqueeze + expand + elementwise.
  int ndim = static_cast<int>(shape.size());
  int d = dim.value();
  if (d < 0)
    d = ndim + d;
  uint32_t reduceSize = shape[d];

  Tensor meanVal = reduce(OperatorEnum::ReduceMean, a, d);
  Tensor meanUnsq = unsqueeze(meanVal, d);
  Tensor meanExp = expand(meanUnsq, shape);
  Tensor diff = binaryOp(OperatorEnum::BinarySub, a, meanExp);
  Tensor diffSq = unaryOp(OperatorEnum::UnarySquare, diff);
  Tensor sumSq = reduce(OperatorEnum::ReduceSum, diffSq, d);
  int denom = static_cast<int>(reduceSize) - correction;
  if (denom > 0) {
    return binaryOp(OperatorEnum::BinaryDiv, sumSq, static_cast<float>(denom));
  }
  return full(computeDimParams(shape, d).outShape, 0.0f, dtype);
}

// =========================================================================
// Softmax
// =========================================================================

Tensor Operations::softmax(const Tensor &a, int dim) {
  auto shape = getShape(a);
  int ndim = static_cast<int>(shape.size());
  if (dim < 0)
    dim = ndim + dim;

  // softmax(x, dim) = exp(x - max(x, dim)) / sum(exp(x - max(x, dim)), dim)
  // All ops stay on GPU via reduce + unsqueeze + expand + elementwise ops.
  Tensor maxVal = reduce(OperatorEnum::ReduceMax, a, dim);
  Tensor maxUnsq = unsqueeze(maxVal, dim);
  Tensor maxExp = expand(maxUnsq, shape);
  Tensor shifted = binaryOp(OperatorEnum::BinarySub, a, maxExp);
  Tensor exps = unaryOp(OperatorEnum::UnaryExp, shifted);
  Tensor sumVal = reduce(OperatorEnum::ReduceSum, exps, dim);
  Tensor sumUnsq = unsqueeze(sumVal, dim);
  Tensor sumExp = expand(sumUnsq, shape);
  return binaryOp(OperatorEnum::BinaryDiv, exps, sumExp);
}

Tensor Operations::logSoftmax(const Tensor &a, int dim) {
  auto shape = getShape(a);
  int ndim = static_cast<int>(shape.size());
  if (dim < 0)
    dim = ndim + dim;

  // logSoftmax(x, dim) = (x - max) - log(sum(exp(x - max), dim))
  Tensor maxVal = reduce(OperatorEnum::ReduceMax, a, dim);
  Tensor maxUnsq = unsqueeze(maxVal, dim);
  Tensor maxExp = expand(maxUnsq, shape);
  Tensor shifted = binaryOp(OperatorEnum::BinarySub, a, maxExp);
  Tensor exps = unaryOp(OperatorEnum::UnaryExp, shifted);
  Tensor sumVal = reduce(OperatorEnum::ReduceSum, exps, dim);
  Tensor logSum = unaryOp(OperatorEnum::UnaryLog, sumVal);
  Tensor logSumUnsq = unsqueeze(logSum, dim);
  Tensor logSumExp = expand(logSumUnsq, shape);
  return binaryOp(OperatorEnum::BinarySub, shifted, logSumExp);
}

// =========================================================================
// Tensor creation
// =========================================================================

Tensor Operations::arange(DataReference start,
                          DataReference end,
                          DataReference step,
                          DataType dtype) {
  if (dtype == DataType::Int32) {
    int32_t s, e, st;
    std::memcpy(&s, start.ptr, sizeof(int32_t));
    std::memcpy(&e, end.ptr, sizeof(int32_t));
    std::memcpy(&st, step.ptr, sizeof(int32_t));
    if (st == 0)
      throw std::runtime_error("step cannot be zero");
    int n = (e - s) / st;
    if (n < 0)
      n = 0;
    std::vector<uint32_t> shape = {static_cast<uint32_t>(n)};
    std::vector<int32_t> values(n);
    for (int i = 0; i < n; ++i)
      values[i] = s + i * st;
    return store_->createTensor(shape, dtype, values.data());
  } else if (dtype == DataType::UInt32) {
    uint32_t s, e, st;
    std::memcpy(&s, start.ptr, sizeof(uint32_t));
    std::memcpy(&e, end.ptr, sizeof(uint32_t));
    std::memcpy(&st, step.ptr, sizeof(uint32_t));
    if (st == 0)
      throw std::runtime_error("step cannot be zero");
    int n = static_cast<int>((e - s) / st);
    if (n < 0)
      n = 0;
    std::vector<uint32_t> shape = {static_cast<uint32_t>(n)};
    std::vector<uint32_t> values(n);
    for (int i = 0; i < n; ++i)
      values[i] = s + i * st;
    return store_->createTensor(shape, dtype, values.data());
  } else {
    float s, e, st;
    std::memcpy(&s, start.ptr, sizeof(float));
    std::memcpy(&e, end.ptr, sizeof(float));
    std::memcpy(&st, step.ptr, sizeof(float));
    if (st == 0.0f)
      throw std::runtime_error("step cannot be zero");
    int n = static_cast<int>((e - s) / st);
    if (n < 0)
      n = 0;
    std::vector<uint32_t> shape = {static_cast<uint32_t>(n)};
    std::vector<float> values(n);
    for (int i = 0; i < n; ++i)
      values[i] = s + i * st;
    return store_->createTensor(shape, dtype, values.data());
  }
}

Tensor Operations::linspace(DataReference start,
                            DataReference end,
                            int steps,
                            DataType dtype) {
  if (steps < 1) {
    throw std::runtime_error("steps must be at least 1");
  }

  float s, e;
  std::memcpy(&s, start.ptr, sizeof(float));
  std::memcpy(&e, end.ptr, sizeof(float));

  std::vector<float> values(steps);
  if (steps == 1) {
    values[0] = s;
  } else {
    float stepSize = (e - s) / (steps - 1);
    for (int i = 0; i < steps; ++i)
      values[i] = s + i * stepSize;
  }

  std::vector<uint32_t> shape = {static_cast<uint32_t>(steps)};
  return store_->createTensor(shape, dtype, values.data());
}

Tensor Operations::full(const std::vector<uint32_t> &shape,
                        DataReference fillValue,
                        DataType dtype) {
  size_t totalSize = shapeProduct(shape);

  if (dtype == DataType::Int32) {
    int32_t val;
    std::memcpy(&val, fillValue.ptr, sizeof(int32_t));
    std::vector<int32_t> values(totalSize, val);
    return store_->createTensor(shape, dtype, values.data());
  } else if (dtype == DataType::UInt32) {
    uint32_t val;
    std::memcpy(&val, fillValue.ptr, sizeof(uint32_t));
    std::vector<uint32_t> values(totalSize, val);
    return store_->createTensor(shape, dtype, values.data());
  } else {
    float val;
    std::memcpy(&val, fillValue.ptr, sizeof(float));
    std::vector<float> values(totalSize, val);
    return store_->createTensor(shape, dtype, values.data());
  }
}

// =========================================================================
// Shape ops (copy data to new buffer with new shape)
// =========================================================================

Tensor Operations::reshape(const Tensor &a,
                           const std::vector<int32_t> &newShape) {
  auto resolved = resolveReshapeShape(getShape(a), newShape);
  auto node = std::make_unique<CopyOpNode>(*store_, a, std::move(resolved));
  return recordOrEncode(std::move(node));
}

Tensor Operations::squeeze(const Tensor &a, std::optional<int> dim) {
  auto oldShape = getShape(a);
  std::vector<uint32_t> newShape;
  int ndim = static_cast<int>(oldShape.size());

  if (!dim.has_value()) {
    // Squeeze all size-1 dims
    for (uint32_t s : oldShape) {
      if (s != 1)
        newShape.push_back(s);
    }
  } else {
    int d = dim.value();
    if (d < 0)
      d = ndim + d;
    if (d < 0 || d >= ndim) {
      throw std::runtime_error("dim " + std::to_string(d) +
                               " out of range for tensor with " +
                               std::to_string(ndim) + " dimensions");
    }
    for (int i = 0; i < ndim; ++i) {
      if (i == d && oldShape[i] == 1)
        continue;
      newShape.push_back(oldShape[i]);
    }
  }

  if (newShape.empty())
    newShape.push_back(1);

  auto node = std::make_unique<CopyOpNode>(*store_, a, std::move(newShape));
  return recordOrEncode(std::move(node));
}

Tensor Operations::unsqueeze(const Tensor &a, int dim) {
  auto oldShape = getShape(a);
  int ndim = static_cast<int>(oldShape.size());

  if (dim < 0)
    dim = ndim + 1 + dim;
  if (dim < 0 || dim > ndim) {
    throw std::runtime_error(
        "dim " + std::to_string(dim) + " out of range for tensor with " +
        std::to_string(ndim) + " dimensions (valid range: [" +
        std::to_string(-(ndim + 1)) + ", " + std::to_string(ndim) + "])");
  }

  std::vector<uint32_t> newShape(oldShape.begin(), oldShape.begin() + dim);
  newShape.push_back(1);
  newShape.insert(newShape.end(), oldShape.begin() + dim, oldShape.end());

  auto node = std::make_unique<CopyOpNode>(*store_, a, std::move(newShape));
  return recordOrEncode(std::move(node));
}

Tensor Operations::unflatten(const Tensor &a,
                             int dim,
                             const std::vector<uint32_t> &sizes) {
  auto oldShape = getShape(a);
  int ndim = static_cast<int>(oldShape.size());
  if (dim < 0)
    dim = ndim + dim;
  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("dim " + std::to_string(dim) +
                                " out of range for tensor with " +
                                std::to_string(ndim) + " dimensions");
  }

  size_t expected = oldShape[dim];
  size_t actual = 1;
  for (uint32_t s : sizes)
    actual *= s;

  if (expected != actual) {
    throw std::invalid_argument("Product of sizes (" + std::to_string(actual) +
                                ") must match dimension " +
                                std::to_string(dim) + " size (" +
                                std::to_string(expected) + ")");
  }

  std::vector<uint32_t> newShape(oldShape.begin(), oldShape.begin() + dim);
  newShape.insert(newShape.end(), sizes.begin(), sizes.end());
  newShape.insert(newShape.end(), oldShape.begin() + dim + 1, oldShape.end());

  auto node = std::make_unique<CopyOpNode>(*store_, a, std::move(newShape));
  return recordOrEncode(std::move(node));
}

Tensor Operations::flatten(const Tensor &a, int startDim, int endDim) {
  auto oldShape = getShape(a);
  int ndim = static_cast<int>(oldShape.size());
  if (ndim == 0) {
    auto node = std::make_unique<CopyOpNode>(*store_, a,
                                             std::vector<uint32_t>(oldShape));
    return recordOrEncode(std::move(node));
  }

  if (endDim < 0)
    endDim = ndim + endDim;
  if (startDim < 0)
    startDim = ndim + startDim;

  if (startDim < 0 || startDim >= ndim)
    throw std::runtime_error("start_dim " + std::to_string(startDim) +
                             " out of range for " + std::to_string(ndim) +
                             "D tensor");
  if (endDim < 0 || endDim >= ndim)
    throw std::runtime_error("end_dim " + std::to_string(endDim) +
                             " out of range for " + std::to_string(ndim) +
                             "D tensor");
  if (startDim > endDim)
    throw std::runtime_error("start_dim must be <= end_dim");

  std::vector<uint32_t> newShape;
  if (startDim == 0 && endDim == ndim - 1) {
    newShape.push_back(static_cast<uint32_t>(shapeProduct(oldShape)));
  } else {
    newShape.insert(newShape.end(), oldShape.begin(),
                    oldShape.begin() + startDim);
    uint32_t flattenedSize = 1;
    for (int i = startDim; i <= endDim; ++i)
      flattenedSize *= oldShape[i];
    newShape.push_back(flattenedSize);
    newShape.insert(newShape.end(), oldShape.begin() + endDim + 1,
                    oldShape.end());
  }

  auto node = std::make_unique<CopyOpNode>(*store_, a, std::move(newShape));
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Norm
// =========================================================================

Tensor Operations::norm(const Tensor &a,
                        std::optional<int> dim,
                        std::optional<uint32_t> spec) {
  if (!dim.has_value()) {
    auto node = std::make_unique<NormOpNode>(*store_, a, spec);
    return recordOrEncode(std::move(node));
  }
  auto node = std::make_unique<DimReduceOpNode>(OperatorEnum::NormDim, *store_,
                                                a, dim.value(), spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Prefix scan
// =========================================================================

Tensor Operations::prefixScan(const Tensor &a,
                              OperatorEnum op,
                              std::optional<uint32_t> spec) {
  auto node = std::make_unique<PrefixScanOpNode>(op, *store_, a, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Sort (in-place)
// =========================================================================

// =========================================================================
// Convolution ops
// =========================================================================

Tensor Operations::conv1d(const Tensor &input,
                          const Tensor &weight,
                          uint32_t stride,
                          uint32_t padding,
                          std::optional<uint32_t> spec) {
  auto node = std::make_unique<Conv1DOpNode>(*store_, input, weight, stride,
                                             padding, spec);
  auto inputs = resolveAndCastInputs(*node, {input, weight});
  if (inputs[0] != input || inputs[1] != weight)
    node = std::make_unique<Conv1DOpNode>(*store_, inputs[0], inputs[1], stride,
                                          padding, spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::conv2d(const Tensor &input,
                          const Tensor &weight,
                          uint32_t strideH,
                          uint32_t strideW,
                          uint32_t padH,
                          uint32_t padW,
                          std::optional<uint32_t> spec) {
  auto node = std::make_unique<Conv2DOpNode>(*store_, input, weight, strideH,
                                             strideW, padH, padW, spec);
  auto inputs = resolveAndCastInputs(*node, {input, weight});
  if (inputs[0] != input || inputs[1] != weight)
    node = std::make_unique<Conv2DOpNode>(*store_, inputs[0], inputs[1],
                                          strideH, strideW, padH, padW, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Pooling ops
// =========================================================================

Tensor Operations::maxPool2d(const Tensor &input,
                             uint32_t kernelH,
                             uint32_t kernelW,
                             uint32_t strideH,
                             uint32_t strideW,
                             uint32_t padH,
                             uint32_t padW,
                             std::optional<uint32_t> spec) {
  auto node = std::make_unique<MaxPool2DOpNode>(
      *store_, input, kernelH, kernelW, strideH, strideW, padH, padW, spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::avgPool2d(const Tensor &input,
                             uint32_t kernelH,
                             uint32_t kernelW,
                             uint32_t strideH,
                             uint32_t strideW,
                             uint32_t padH,
                             uint32_t padW,
                             std::optional<uint32_t> spec) {
  auto node = std::make_unique<AvgPool2DOpNode>(
      *store_, input, kernelH, kernelW, strideH, strideW, padH, padW, spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::adaptiveAvgPool2d(const Tensor &input,
                                     uint32_t outH,
                                     uint32_t outW,
                                     std::optional<uint32_t> spec) {
  auto inShape = getShape(input);
  if (inShape.size() != 4)
    throw std::runtime_error(
        "adaptive_avg_pool2d: input must be 4D [N, C, H, W]");

  uint32_t H_in = inShape[2], W_in = inShape[3];

  uint32_t sH = H_in / outH;
  uint32_t sW = W_in / outW;
  uint32_t kH = H_in - (outH - 1) * sH;
  uint32_t kW = W_in - (outW - 1) * sW;

  return avgPool2d(input, kH, kW, sH, sW, 0, 0, spec);
}

// =========================================================================
// Normalization ops
// =========================================================================

Tensor Operations::layerNorm(const Tensor &input,
                             const std::vector<uint32_t> &normalizedShape,
                             const Tensor *weight,
                             const Tensor *bias,
                             float eps) {
  auto shape = getShape(input);
  auto dtype = getDtype(input);

  // normalizedShape must match the trailing dimensions of input
  int ndim = static_cast<int>(shape.size());
  int normDims = static_cast<int>(normalizedShape.size());
  int axisDim = ndim - normDims;

  if (axisDim < 0) {
    throw std::runtime_error(
        "layer_norm: normalizedShape has more dims than input");
  }
  for (int i = 0; i < normDims; ++i) {
    if (shape[axisDim + i] != normalizedShape[i]) {
      throw std::runtime_error(
          "layer_norm: normalizedShape doesn't match input trailing dims");
    }
  }

  // Compute sizes
  size_t outerSize = 1;
  for (int i = 0; i < axisDim; ++i)
    outerSize *= shape[i];

  size_t normSize = 1;
  for (int i = 0; i < normDims; ++i)
    normSize *= normalizedShape[i];

  // All ops on GPU via reduce + unsqueeze + expand + elementwise.
  // Flatten to [outer, norm] so we can reduce along dim=1.
  Tensor flat = reshape(
      input, {static_cast<int32_t>(outerSize), static_cast<int32_t>(normSize)});
  std::vector<uint32_t> flatShape = {static_cast<uint32_t>(outerSize),
                                     static_cast<uint32_t>(normSize)};

  // mean and variance along the norm dimension (dim=1)
  Tensor meanVal = reduce(OperatorEnum::ReduceMean, flat, 1);
  Tensor meanUnsq = unsqueeze(meanVal, 1);
  Tensor meanExp = expand(meanUnsq, flatShape);
  Tensor diff = binaryOp(OperatorEnum::BinarySub, flat, meanExp);
  Tensor diffSq = unaryOp(OperatorEnum::UnarySquare, diff);
  Tensor varVal = reduce(OperatorEnum::ReduceMean, diffSq, 1);
  Tensor varEps = binaryOp(OperatorEnum::BinaryAdd, varVal, eps);
  Tensor invStd = unaryOp(OperatorEnum::UnaryRsqrt, varEps);
  Tensor invStdUnsq = unsqueeze(invStd, 1);
  Tensor invStdExp = expand(invStdUnsq, flatShape);

  // normalized = (x - mean) * invStd
  Tensor normalized = binaryOp(OperatorEnum::BinaryMul, diff, invStdExp);

  // Apply weight and bias if provided
  if (weight) {
    Tensor wFlat = reshape(*weight, {1, static_cast<int32_t>(normSize)});
    Tensor wExp = expand(wFlat, flatShape);
    normalized = binaryOp(OperatorEnum::BinaryMul, normalized, wExp);
  }
  if (bias) {
    Tensor bFlat = reshape(*bias, {1, static_cast<int32_t>(normSize)});
    Tensor bExp = expand(bFlat, flatShape);
    normalized = binaryOp(OperatorEnum::BinaryAdd, normalized, bExp);
  }

  // Reshape back to original shape
  std::vector<int32_t> origShape(shape.begin(), shape.end());
  return reshape(normalized, origShape);
}

Tensor Operations::batchNorm(const Tensor &input,
                             const Tensor &runningMean,
                             const Tensor &runningVar,
                             const Tensor *weight,
                             const Tensor *bias,
                             float eps) {
  auto shape = getShape(input);
  auto dtype = getDtype(input);

  if (shape.size() < 2) {
    throw std::runtime_error(
        "batch_norm: input must be at least 2D [N, C, ...]");
  }

  uint32_t N = shape[0];
  uint32_t C = shape[1];
  uint32_t spatialSize = 1;
  for (size_t i = 2; i < shape.size(); ++i)
    spatialSize *= shape[i];

  // All ops on GPU via reshape + expand + elementwise.
  // Flatten input to [N, C, spatial] so per-channel params broadcast easily.
  std::vector<uint32_t> flatShape = {N, C, spatialSize};
  Tensor flat =
      reshape(input, {static_cast<int32_t>(N), static_cast<int32_t>(C),
                      static_cast<int32_t>(spatialSize)});

  // Reshape per-channel vectors [C] -> [1, C, 1] then expand to [N, C, spatial]
  std::vector<int32_t> chanShape = {1, static_cast<int32_t>(C), 1};

  // invStd = rsqrt(runningVar + eps)
  Tensor varEps = binaryOp(OperatorEnum::BinaryAdd, runningVar, eps);
  Tensor invStd = unaryOp(OperatorEnum::UnaryRsqrt, varEps);

  // scale = weight * invStd  (or just invStd if no weight)
  Tensor scale =
      weight ? binaryOp(OperatorEnum::BinaryMul, *weight, invStd) : invStd;

  // shift = bias - runningMean * scale  (or -runningMean * scale)
  Tensor meanScale = binaryOp(OperatorEnum::BinaryMul, runningMean, scale);
  Tensor shift = bias ? binaryOp(OperatorEnum::BinarySub, *bias, meanScale)
                      : unaryOp(OperatorEnum::UnaryNeg, meanScale);

  // Broadcast scale and shift to match [N, C, spatial]
  Tensor scaleExp = expand(reshape(scale, chanShape), flatShape);
  Tensor shiftExp = expand(reshape(shift, chanShape), flatShape);

  // result = input * scale + shift
  Tensor scaled = binaryOp(OperatorEnum::BinaryMul, flat, scaleExp);
  Tensor result = binaryOp(OperatorEnum::BinaryAdd, scaled, shiftExp);

  // Reshape back to original shape
  std::vector<int32_t> origShape(shape.begin(), shape.end());
  return reshape(result, origShape);
}

Tensor Operations::rmsNorm(const Tensor &x,
                           const Tensor &weight,
                           float eps,
                           std::optional<uint32_t> spec) {
  auto node = std::make_unique<RMSNormOpNode>(*store_, x, weight, eps, spec);
  return recordOrEncode(std::move(node));
}

Tensor Operations::extendedRmsNorm(const Tensor &residual_base,
                                   const Tensor &delta,
                                   const Tensor &weight,
                                   float eps,
                                   std::optional<uint32_t> spec) {
  auto node = std::make_unique<ExtendedRMSNormOpNode>(*store_, residual_base,
                                                      delta, weight, eps, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Embedding ops
// =========================================================================

Tensor Operations::embedding(const Tensor &indices,
                             const Tensor &weight,
                             std::optional<uint32_t> spec) {
  auto node = std::make_unique<EmbeddingOpNode>(*store_, indices, weight, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Padding ops
// =========================================================================

Tensor Operations::pad(const Tensor &input,
                       const std::vector<uint32_t> &padWidths,
                       float value,
                       std::optional<uint32_t> spec) {
  auto node = std::make_unique<PadOpNode>(
      *store_, input, std::vector<uint32_t>(padWidths), value, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// RoPE
// =========================================================================

Tensor Operations::applyRoPE(const Tensor &x,
                             const Tensor &cosTable,
                             const Tensor &sinTable,
                             const Tensor &runtimeParams,
                             uint32_t headDim,
                             const Tensor &preallocOutput,
                             std::optional<uint32_t> spec) {
  auto node = std::make_unique<RoPEOpNode>(*store_, x, cosTable, sinTable,
                                           runtimeParams, headDim,
                                           preallocOutput, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Attention
// =========================================================================

void Operations::cacheWrite(const Tensor &cache,
                            const Tensor &newData,
                            const Tensor &runtimeParams) {
  auto node = std::make_unique<CacheWriteOpNode>(*store_, newData,
                                                 runtimeParams, cache);
  dispatch(std::move(node));
}

Tensor Operations::attention(const Tensor &q,
                             const Tensor &kCache,
                             const Tensor &vCache,
                             const Tensor &runtimeParams,
                             uint32_t nHeads,
                             uint32_t nKvHeads,
                             uint32_t headDim,
                             const Tensor &preallocOutput,
                             std::optional<uint32_t> spec) {
  auto node = std::make_unique<AttentionOpNode>(*store_, q, kCache, vCache,
                                                runtimeParams, nHeads, nKvHeads,
                                                headDim, preallocOutput, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Expand
// =========================================================================

Tensor Operations::expand(const Tensor &a,
                          const std::vector<uint32_t> &targetShape,
                          std::optional<uint32_t> spec) {
  auto node = std::make_unique<ExpandOpNode>(*store_, a, targetShape, spec);
  return recordOrEncode(std::move(node));
}

// =========================================================================
// Dtype resolution helper
// =========================================================================

std::vector<Tensor>
Operations::resolveAndCastInputs(const OpNode &node,
                                 const std::vector<Tensor> &inputs) {
  std::vector<DataType> actual;
  for (const auto &t : inputs)
    actual.push_back(getDtype(t));

  auto resolved = node.resolveInputDtypes(actual);
  if (resolved == actual)
    return inputs;

  std::vector<Tensor> result;
  for (size_t i = 0; i < inputs.size(); ++i) {
    result.push_back(resolved[i] != actual[i] ? cast(inputs[i], resolved[i])
                                              : inputs[i]);
  }
  return result;
}

// =========================================================================
// Type conversion
// =========================================================================

Tensor Operations::dequantize(const Tensor &rawData,
                              uint32_t format,
                              uint32_t rows,
                              uint32_t cols) {
  auto node = std::make_unique<DequantOpNode>(
      *store_, rawData, static_cast<DequantFormat>(format), rows, cols);
  return recordOrEncode(std::move(node));
}

Tensor Operations::cast(const Tensor &input, DataType targetDtype) {
  if (getDtype(input) == targetDtype)
    return input;

  // Return cached result if we already cast this tensor in the current graph
  for (const auto &entry : castCache_) {
    if (entry.input == input && entry.targetDtype == targetDtype)
      return entry.output;
  }

  auto node = std::make_unique<CastOpNode>(*store_, input, targetDtype);
  Tensor output = recordOrEncode(std::move(node));
  castCache_.push_back({input, targetDtype, output});
  return output;
}

// =========================================================================
// Sort (in-place)
// =========================================================================

void Operations::sortBitonic(const Tensor &keys,
                             const Tensor &vals,
                             std::optional<uint32_t> spec) {
  flush();
  runtime_->dispatch(
      std::make_unique<BitonicSortOpNode>(*store_, keys, vals, spec));
}

void Operations::sortRadix(const Tensor &keys,
                           const Tensor &vals,
                           std::optional<uint32_t> spec) {
  flush();
  runtime_->dispatch(
      std::make_unique<RadixSortOpNode>(*store_, keys, vals, spec));
}

} // namespace cut
