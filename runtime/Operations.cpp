#include "Operations.h"
#include "OpNode.h"
#include "Runtime.h"
#include "ShapeUtils.h"

#include "impl/avgpool2d/AvgPool2DOp.h"
#include "impl/binary/BinaryOp.h"
#include "impl/conv1d/Conv1DOp.h"
#include "impl/conv2d/Conv2DOp.h"
#include "impl/matmul/MatMulOp.h"
#include "impl/maxpool2d/MaxPool2DOp.h"
#include "impl/memory/MemoryOp.h"
#include "impl/reduce/ReduceOp.h"
#include "impl/reducedim/ReduceDimOp.h"
#include "impl/scan/ScanOp.h"
#include "impl/sort/SortOp.h"
#include "impl/ternary/TernaryOp.h"
#include "impl/transpose/TransposeOp.h"
#include "impl/unary/UnaryOp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cut {

using graph::Graph;

// =========================================================================
// Operations
// =========================================================================

Operations::Operations(Runtime &runtime) : runtime_(&runtime) {}

std::vector<uint32_t> Operations::getShape(const Tensor &h) const {
  return runtime_->getTensor(h).getShape();
}

DataType Operations::getDtype(const Tensor &h) const {
  return runtime_->getTensor(h).getDtype();
}

// =========================================================================
// Graph mode helpers
// =========================================================================

void Operations::setGraph(Graph *g) {
  graph_ = g;
  tensorToNodeId_.clear();
}

void Operations::clearGraph() {
  graph_ = nullptr;
  tensorToNodeId_.clear();
}

bool Operations::isGraphMode() const {
  return graph_ != nullptr;
}

const std::vector<std::pair<Tensor, uint32_t>> &
Operations::graphMappings() const {
  return tensorToNodeId_;
}

uint32_t Operations::toNodeId(const Tensor &t) {
  // Check if already mapped
  for (const auto &p : tensorToNodeId_) {
    if (p.first == t)
      return p.second;
  }
  // Register as a new graph input
  registerInput(t, false);
  return tensorToNodeId_.back().second;
}

Tensor Operations::registerInput(const Tensor &gpuHandle, bool isConstant) {
  // Deduplicate only constant inputs (weights). Dynamic inputs may reuse the
  // same placeholder handle for distinct graph inputs that receive different
  // tensors at execution time, so each call must create a separate node.
  if (isConstant) {
    for (const auto &p : tensorToNodeId_) {
      if (p.first == gpuHandle)
        return gpuHandle;
    }
  }
  auto shape = getShape(gpuHandle);
  auto dtype = getDtype(gpuHandle);
  auto node =
      std::make_unique<InputOpNode>(gpuHandle, shape, dtype, isConstant);
  uint32_t nodeId = graph_->addNode(std::move(node), gpuHandle);
  tensorToNodeId_.emplace_back(gpuHandle, nodeId);
  return gpuHandle;
}

Tensor Operations::recordOrEncode(std::unique_ptr<OpNode> node,
                                  const std::vector<Tensor> &inputs) {
  Tensor output = node->output();
  if (graph_) {
    std::vector<uint32_t> inputIds;
    for (const auto &inp : inputs)
      inputIds.push_back(toNodeId(inp));
    node->setGraphInputIds(std::move(inputIds));
    uint32_t nodeId = graph_->addNode(std::move(node), output);
    tensorToNodeId_.emplace_back(output, nodeId);
    return output;
  }
  runtime_->encodeOperator(std::move(node));
  return output;
}

void Operations::encodeOp(std::unique_ptr<OpNode> node) {
  runtime_->encodeOperator(std::move(node));
}

void Operations::encodeOp(OpNode &node) {
  runtime_->encodeOperator(node);
}

// =========================================================================
// Generic element-wise ops
// =========================================================================

Tensor Operations::binaryOp(OperatorEnum op,
                            const Tensor &a,
                            const Tensor &b,
                            std::optional<uint32_t> spec) {
  auto node = std::make_unique<BinaryVecVecOpNode>(op, *runtime_, a, b, spec);
  return recordOrEncode(std::move(node), {a, b});
}

Tensor Operations::unaryOp(OperatorEnum op,
                           const Tensor &a,
                           std::optional<uint32_t> spec) {
  auto node = std::make_unique<UnaryOpNode>(op, *runtime_, a, spec);
  return recordOrEncode(std::move(node), {a});
}

Tensor Operations::vecScalarOp(OperatorEnum op,
                               const Tensor &a,
                               DataReference scalar,
                               std::optional<uint32_t> spec) {
  uint32_t scalarBits = 0;
  std::memcpy(&scalarBits, scalar.ptr, sizeof(uint32_t));
  auto node = std::make_unique<BinaryVecScalarOpNode>(op, *runtime_, a,
                                                      scalarBits, spec);
  return recordOrEncode(std::move(node), {a});
}

// =========================================================================
// Reduction ops
// =========================================================================

Tensor Operations::reduce(OperatorEnum op,
                          const Tensor &a,
                          std::optional<int> dim,
                          std::optional<uint32_t> spec) {
  if (!dim.has_value()) {
    auto node = std::make_unique<GlobalReduceOpNode>(op, *runtime_, a, spec);
    return recordOrEncode(std::move(node), {a});
  }
  auto node =
      std::make_unique<DimReduceOpNode>(op, *runtime_, a, dim.value(), spec);
  return recordOrEncode(std::move(node), {a});
}

// =========================================================================
// Matrix ops
// =========================================================================

Tensor Operations::matmul(const Tensor &a,
                          const Tensor &b,
                          std::optional<uint32_t> spec) {
  auto node = std::make_unique<MatMulOpNode>(*runtime_, a, b, spec);
  return recordOrEncode(std::move(node), {a, b});
}

Tensor Operations::transpose(const Tensor &a, std::optional<uint32_t> spec) {
  auto node = std::make_unique<TransposeOpNode>(*runtime_, a, spec);
  return recordOrEncode(std::move(node), {a});
}

Tensor Operations::dot(const Tensor &a,
                       const Tensor &b,
                       std::optional<uint32_t> spec) {
  if (graph_) {
    // In graph mode, decompose dot into multiply + reduce
    Tensor mul = binaryOp(BinaryVecVecMul, a, b);
    Tensor sum = reduce(ReduceSum, mul);
    return sum;
  }
  auto node = std::make_unique<DotOpNode>(*runtime_, a, b, spec);
  Tensor partials = node->output();
  uint32_t numWorkgroups = node->outputShape()[0];
  runtime_->encodeOperator(std::move(node));

  // Read back per-workgroup partial sums and accumulate on CPU
  std::vector<float> partialData(numWorkgroups);
  runtime_->copyFromTensor(partials, partialData.data(),
                           numWorkgroups * sizeof(float));
  float result = 0.0f;
  for (uint32_t i = 0; i < numWorkgroups; ++i) {
    result += partialData[i];
  }

  Tensor out = runtime_->createTensorEmpty({1}, DataType::Float32);
  runtime_->copyToTensor(out, &result, sizeof(float));
  return out;
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
  auto node = std::make_unique<TernaryClampOpNode>(*runtime_, a, minBits,
                                                   maxBits, spec);
  return recordOrEncode(std::move(node), {a});
}

Tensor Operations::where(const Tensor &cond,
                         const Tensor &x,
                         const Tensor &y,
                         std::optional<uint32_t> spec) {
  auto node =
      std::make_unique<TernarySelectOpNode>(*runtime_, cond, x, y, spec);
  return recordOrEncode(std::move(node), {cond, x, y});
}

// =========================================================================
// Cumulative ops
// =========================================================================

Tensor Operations::cumOp(const Tensor &a,
                         OperatorEnum op,
                         std::optional<int> dim,
                         std::optional<uint32_t> spec) {
  int d = dim.value_or(0);
  auto node = std::make_unique<CumOpNode>(op, *runtime_, a, d, spec);
  return recordOrEncode(std::move(node), {a});
}

// =========================================================================
// Statistical ops
// =========================================================================

Tensor
Operations::variance(const Tensor &a, int correction, std::optional<int> dim) {
  if (graph_) {
    uint32_t inputNodeId = toNodeId(a);
    auto shape = getShape(a);
    auto dtype = getDtype(a);
    std::vector<uint32_t> outShape;
    if (!dim.has_value()) {
      outShape = {1};
    } else {
      auto params = computeDimParams(shape, dim.value());
      outShape = params.outShape;
    }
    int capturedCorrection = correction;
    auto capturedDim = dim;
    auto node = std::make_unique<DeferredOpNode>(
        outShape, dtype, "Variance",
        [capturedCorrection, capturedDim](Operations &ops,
                                          const std::vector<Tensor> &in) {
          return ops.variance(in[0], capturedCorrection, capturedDim);
        });
    node->setGraphInputIds({inputNodeId});
    Tensor output = runtime_->createTensorEmpty(outShape, dtype);
    uint32_t nodeId = graph_->addNode(std::move(node), output);
    tensorToNodeId_.emplace_back(output, nodeId);
    return output;
  }
  auto shape = getShape(a);
  auto dtype = getDtype(a);

  if (!dim.has_value()) {
    // Global variance
    Tensor sumTensor = reduce(OperatorEnum::ReduceSum, a);
    float total = 0.0f;
    runtime_->copyFromTensor(sumTensor, &total, sizeof(float));
    size_t n = shapeProduct(shape);
    float m = total / static_cast<float>(n);

    std::vector<float> data(n);
    runtime_->copyFromTensor(a, data.data(), n * sizeof(float));

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double diff = static_cast<double>(data[i]) - static_cast<double>(m);
      sum += diff * diff;
    }

    int denom = static_cast<int>(n) - correction;
    float result = denom > 0 ? static_cast<float>(sum / denom) : 0.0f;

    Tensor out = runtime_->createTensorEmpty({1}, dtype);
    runtime_->copyToTensor(out, &result, sizeof(float));
    return out;
  }

  // Dimension-wise variance
  auto params = computeDimParams(shape, dim.value());

  Tensor meanHandle = reduce(OperatorEnum::ReduceMean, a, dim.value());

  size_t totalElements = shapeProduct(shape);
  size_t meanElements = params.outerSize * params.innerSize;

  std::vector<float> flatData(totalElements);
  std::vector<float> meanData(meanElements);

  runtime_->copyFromTensor(a, flatData.data(), totalElements * sizeof(float));
  runtime_->copyFromTensor(meanHandle, meanData.data(),
                           meanElements * sizeof(float));

  std::vector<float> result(meanElements);
  for (uint32_t o = 0; o < params.outerSize; ++o) {
    for (uint32_t iInner = 0; iInner < params.innerSize; ++iInner) {
      float meanVal = meanData[o * params.innerSize + iInner];
      double s = 0.0;
      for (uint32_t r = 0; r < params.reduceSize; ++r) {
        size_t idx =
            static_cast<size_t>(o) * params.reduceSize * params.innerSize +
            r * params.innerSize + iInner;
        double diff =
            static_cast<double>(flatData[idx]) - static_cast<double>(meanVal);
        s += diff * diff;
      }
      int n = static_cast<int>(params.reduceSize) - correction;
      result[o * params.innerSize + iInner] =
          n > 0 ? static_cast<float>(s / n) : 0.0f;
    }
  }

  Tensor out = runtime_->createTensorEmpty(params.outShape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
}

// =========================================================================
// Softmax
// =========================================================================

Tensor Operations::softmax(const Tensor &a, int dim) {
  if (graph_) {
    uint32_t inputNodeId = toNodeId(a);
    auto shape = getShape(a);
    auto dtype = getDtype(a);
    int capturedDim = dim;
    auto node = std::make_unique<DeferredOpNode>(
        shape, dtype, "Softmax",
        [capturedDim](Operations &ops, const std::vector<Tensor> &in) {
          return ops.softmax(in[0], capturedDim);
        });
    node->setGraphInputIds({inputNodeId});
    Tensor output = runtime_->createTensorEmpty(shape, dtype);
    uint32_t nodeId = graph_->addNode(std::move(node), output);
    tensorToNodeId_.emplace_back(output, nodeId);
    return output;
  }
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  int ndim = static_cast<int>(shape.size());
  if (dim < 0)
    dim = ndim + dim;

  auto params = computeDimParams(shape, dim);

  // Compute dim-wise max for numerical stability (on GPU)
  Tensor maxHandle = reduce(OperatorEnum::ReduceMax, a, dim);

  // Read data from GPU
  size_t totalElements = shapeProduct(shape);
  size_t maxElements = params.outerSize * params.innerSize;

  std::vector<float> flatData(totalElements);
  std::vector<float> maxData(maxElements);

  runtime_->copyFromTensor(a, flatData.data(), totalElements * sizeof(float));
  runtime_->copyFromTensor(maxHandle, maxData.data(),
                           maxElements * sizeof(float));

  // Compute softmax on CPU
  std::vector<float> result(totalElements);
  for (uint32_t o = 0; o < params.outerSize; ++o) {
    for (uint32_t iInner = 0; iInner < params.innerSize; ++iInner) {
      float maxVal = maxData[o * params.innerSize + iInner];

      // Compute exp(x - max) and sum
      float expSum = 0.0f;
      for (uint32_t r = 0; r < params.reduceSize; ++r) {
        size_t idx =
            static_cast<size_t>(o) * params.reduceSize * params.innerSize +
            r * params.innerSize + iInner;
        float e = std::exp(flatData[idx] - maxVal);
        result[idx] = e;
        expSum += e;
      }

      // Normalize
      for (uint32_t r = 0; r < params.reduceSize; ++r) {
        size_t idx =
            static_cast<size_t>(o) * params.reduceSize * params.innerSize +
            r * params.innerSize + iInner;
        result[idx] /= expSum;
      }
    }
  }

  Tensor out = runtime_->createTensorEmpty(shape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
}

Tensor Operations::logSoftmax(const Tensor &a, int dim) {
  if (graph_) {
    uint32_t inputNodeId = toNodeId(a);
    auto shape = getShape(a);
    auto dtype = getDtype(a);
    int capturedDim = dim;
    auto node = std::make_unique<DeferredOpNode>(
        shape, dtype, "LogSoftmax",
        [capturedDim](Operations &ops, const std::vector<Tensor> &in) {
          return ops.logSoftmax(in[0], capturedDim);
        });
    node->setGraphInputIds({inputNodeId});
    Tensor output = runtime_->createTensorEmpty(shape, dtype);
    uint32_t nodeId = graph_->addNode(std::move(node), output);
    tensorToNodeId_.emplace_back(output, nodeId);
    return output;
  }
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  int ndim = static_cast<int>(shape.size());
  if (dim < 0)
    dim = ndim + dim;

  auto params = computeDimParams(shape, dim);

  // Compute dim-wise max for numerical stability (on GPU)
  Tensor maxHandle = reduce(OperatorEnum::ReduceMax, a, dim);

  // Read data from GPU
  size_t totalElements = shapeProduct(shape);
  size_t maxElements = params.outerSize * params.innerSize;

  std::vector<float> flatData(totalElements);
  std::vector<float> maxData(maxElements);

  runtime_->copyFromTensor(a, flatData.data(), totalElements * sizeof(float));
  runtime_->copyFromTensor(maxHandle, maxData.data(),
                           maxElements * sizeof(float));

  // Compute log softmax on CPU
  std::vector<float> result(totalElements);
  for (uint32_t o = 0; o < params.outerSize; ++o) {
    for (uint32_t iInner = 0; iInner < params.innerSize; ++iInner) {
      float maxVal = maxData[o * params.innerSize + iInner];

      float expSum = 0.0f;
      for (uint32_t r = 0; r < params.reduceSize; ++r) {
        size_t idx =
            static_cast<size_t>(o) * params.reduceSize * params.innerSize +
            r * params.innerSize + iInner;
        expSum += std::exp(flatData[idx] - maxVal);
      }
      float logSum = std::log(expSum) + maxVal;

      for (uint32_t r = 0; r < params.reduceSize; ++r) {
        size_t idx =
            static_cast<size_t>(o) * params.reduceSize * params.innerSize +
            r * params.innerSize + iInner;
        result[idx] = flatData[idx] - logSum;
      }
    }
  }

  Tensor out = runtime_->createTensorEmpty(shape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
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
    return runtime_->createTensor(shape, dtype, values.data());
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
    return runtime_->createTensor(shape, dtype, values.data());
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
    return runtime_->createTensor(shape, dtype, values.data());
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
  return runtime_->createTensor(shape, dtype, values.data());
}

Tensor Operations::full(const std::vector<uint32_t> &shape,
                        DataReference fillValue,
                        DataType dtype) {
  size_t totalSize = shapeProduct(shape);

  if (dtype == DataType::Int32) {
    int32_t val;
    std::memcpy(&val, fillValue.ptr, sizeof(int32_t));
    std::vector<int32_t> values(totalSize, val);
    return runtime_->createTensor(shape, dtype, values.data());
  } else if (dtype == DataType::UInt32) {
    uint32_t val;
    std::memcpy(&val, fillValue.ptr, sizeof(uint32_t));
    std::vector<uint32_t> values(totalSize, val);
    return runtime_->createTensor(shape, dtype, values.data());
  } else {
    float val;
    std::memcpy(&val, fillValue.ptr, sizeof(float));
    std::vector<float> values(totalSize, val);
    return runtime_->createTensor(shape, dtype, values.data());
  }
}

// =========================================================================
// Shape ops (copy data to new buffer with new shape)
// =========================================================================

Tensor Operations::reshape(const Tensor &a,
                           const std::vector<int32_t> &newShape) {
  auto resolved = resolveReshapeShape(getShape(a), newShape);
  auto node = std::make_unique<CopyOpNode>(*runtime_, a, std::move(resolved));
  return recordOrEncode(std::move(node), {a});
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

  auto node = std::make_unique<CopyOpNode>(*runtime_, a, std::move(newShape));
  return recordOrEncode(std::move(node), {a});
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

  auto node = std::make_unique<CopyOpNode>(*runtime_, a, std::move(newShape));
  return recordOrEncode(std::move(node), {a});
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

  auto node = std::make_unique<CopyOpNode>(*runtime_, a, std::move(newShape));
  return recordOrEncode(std::move(node), {a});
}

Tensor Operations::flatten(const Tensor &a, int startDim, int endDim) {
  auto oldShape = getShape(a);
  int ndim = static_cast<int>(oldShape.size());
  if (ndim == 0) {
    auto node = std::make_unique<CopyOpNode>(*runtime_, a,
                                             std::vector<uint32_t>(oldShape));
    return recordOrEncode(std::move(node), {a});
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

  auto node = std::make_unique<CopyOpNode>(*runtime_, a, std::move(newShape));
  return recordOrEncode(std::move(node), {a});
}

// =========================================================================
// Norm
// =========================================================================

Tensor Operations::norm(const Tensor &a,
                        std::optional<int> dim,
                        std::optional<uint32_t> spec) {
  if (!dim.has_value()) {
    auto node = std::make_unique<NormOpNode>(*runtime_, a, spec);
    return recordOrEncode(std::move(node), {a});
  }
  auto node = std::make_unique<DimReduceOpNode>(
      OperatorEnum::NormDim, *runtime_, a, dim.value(), spec);
  return recordOrEncode(std::move(node), {a});
}

// =========================================================================
// Prefix scan
// =========================================================================

Tensor Operations::prefixScan(const Tensor &a,
                              OperatorEnum op,
                              std::optional<uint32_t> spec) {
  auto node = std::make_unique<PrefixScanOpNode>(op, *runtime_, a, spec);
  return recordOrEncode(std::move(node), {a});
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
  auto node = std::make_unique<Conv1DOpNode>(*runtime_, input, weight, stride,
                                             padding, spec);
  return recordOrEncode(std::move(node), {input, weight});
}

Tensor Operations::conv2d(const Tensor &input,
                          const Tensor &weight,
                          uint32_t strideH,
                          uint32_t strideW,
                          uint32_t padH,
                          uint32_t padW,
                          std::optional<uint32_t> spec) {
  auto node = std::make_unique<Conv2DOpNode>(*runtime_, input, weight, strideH,
                                             strideW, padH, padW, spec);
  return recordOrEncode(std::move(node), {input, weight});
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
      *runtime_, input, kernelH, kernelW, strideH, strideW, padH, padW, spec);
  return recordOrEncode(std::move(node), {input});
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
      *runtime_, input, kernelH, kernelW, strideH, strideW, padH, padW, spec);
  return recordOrEncode(std::move(node), {input});
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
  if (graph_) {
    uint32_t inputNodeId = toNodeId(input);
    auto shape = getShape(input);
    auto dtype = getDtype(input);
    std::vector<uint32_t> inputIds = {inputNodeId};
    if (weight)
      inputIds.push_back(toNodeId(*weight));
    if (bias)
      inputIds.push_back(toNodeId(*bias));

    auto capturedNormShape = normalizedShape;
    float capturedEps = eps;
    bool hasWeight = weight != nullptr;
    bool hasBias = bias != nullptr;
    auto node = std::make_unique<DeferredOpNode>(
        shape, dtype, "LayerNorm",
        [capturedNormShape, capturedEps, hasWeight,
         hasBias](Operations &ops, const std::vector<Tensor> &in) {
          const Tensor *wPtr = nullptr;
          const Tensor *bPtr = nullptr;
          size_t nextInput = 1;
          if (hasWeight)
            wPtr = &in[nextInput++];
          if (hasBias)
            bPtr = &in[nextInput++];
          return ops.layerNorm(in[0], capturedNormShape, wPtr, bPtr,
                               capturedEps);
        });
    node->setGraphInputIds(std::move(inputIds));
    Tensor output = runtime_->createTensorEmpty(shape, dtype);
    uint32_t nodeId = graph_->addNode(std::move(node), output);
    tensorToNodeId_.emplace_back(output, nodeId);
    return output;
  }
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

  size_t totalElements = outerSize * normSize;

  // Read input data
  std::vector<float> data(totalElements);
  runtime_->copyFromTensor(input, data.data(), totalElements * sizeof(float));

  // Read weight/bias if provided
  std::vector<float> wData, bData;
  if (weight) {
    wData.resize(normSize);
    runtime_->copyFromTensor(*weight, wData.data(), normSize * sizeof(float));
  }
  if (bias) {
    bData.resize(normSize);
    runtime_->copyFromTensor(*bias, bData.data(), normSize * sizeof(float));
  }

  // Compute layer norm on CPU
  std::vector<float> result(totalElements);
  for (size_t o = 0; o < outerSize; ++o) {
    size_t base = o * normSize;

    // Mean
    double sum = 0.0;
    for (size_t i = 0; i < normSize; ++i)
      sum += data[base + i];
    float mean = static_cast<float>(sum / normSize);

    // Variance
    double varSum = 0.0;
    for (size_t i = 0; i < normSize; ++i) {
      double diff = data[base + i] - mean;
      varSum += diff * diff;
    }
    float invStd =
        1.0f / std::sqrt(static_cast<float>(varSum / normSize) + eps);

    // Normalize, scale, shift
    for (size_t i = 0; i < normSize; ++i) {
      float normalized = (data[base + i] - mean) * invStd;
      if (weight)
        normalized *= wData[i];
      if (bias)
        normalized += bData[i];
      result[base + i] = normalized;
    }
  }

  Tensor out = runtime_->createTensorEmpty(shape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
}

Tensor Operations::batchNorm(const Tensor &input,
                             const Tensor &runningMean,
                             const Tensor &runningVar,
                             const Tensor *weight,
                             const Tensor *bias,
                             float eps) {
  if (graph_) {
    uint32_t inputNodeId = toNodeId(input);
    uint32_t meanNodeId = toNodeId(runningMean);
    uint32_t varNodeId = toNodeId(runningVar);
    auto shape = getShape(input);
    auto dtype = getDtype(input);
    std::vector<uint32_t> inputIds = {inputNodeId, meanNodeId, varNodeId};
    if (weight)
      inputIds.push_back(toNodeId(*weight));
    if (bias)
      inputIds.push_back(toNodeId(*bias));

    float capturedEps = eps;
    bool hasWeight = weight != nullptr;
    bool hasBias = bias != nullptr;
    auto node = std::make_unique<DeferredOpNode>(
        shape, dtype, "BatchNorm",
        [capturedEps, hasWeight, hasBias](Operations &ops,
                                          const std::vector<Tensor> &in) {
          const Tensor *wPtr = nullptr;
          const Tensor *bPtr = nullptr;
          size_t nextInput = 3;
          if (hasWeight)
            wPtr = &in[nextInput++];
          if (hasBias)
            bPtr = &in[nextInput++];
          return ops.batchNorm(in[0], in[1], in[2], wPtr, bPtr, capturedEps);
        });
    node->setGraphInputIds(std::move(inputIds));
    Tensor output = runtime_->createTensorEmpty(shape, dtype);
    uint32_t nodeId = graph_->addNode(std::move(node), output);
    tensorToNodeId_.emplace_back(output, nodeId);
    return output;
  }
  auto shape = getShape(input);
  auto dtype = getDtype(input);

  if (shape.size() < 2) {
    throw std::runtime_error(
        "batch_norm: input must be at least 2D [N, C, ...]");
  }

  uint32_t N = shape[0];
  uint32_t C = shape[1];
  size_t spatialSize = 1;
  for (size_t i = 2; i < shape.size(); ++i)
    spatialSize *= shape[i];

  size_t totalElements = N * C * spatialSize;

  // Read all data
  std::vector<float> data(totalElements);
  runtime_->copyFromTensor(input, data.data(), totalElements * sizeof(float));

  std::vector<float> meanData(C), varData(C);
  runtime_->copyFromTensor(runningMean, meanData.data(), C * sizeof(float));
  runtime_->copyFromTensor(runningVar, varData.data(), C * sizeof(float));

  std::vector<float> wData, bData;
  if (weight) {
    wData.resize(C);
    runtime_->copyFromTensor(*weight, wData.data(), C * sizeof(float));
  }
  if (bias) {
    bData.resize(C);
    runtime_->copyFromTensor(*bias, bData.data(), C * sizeof(float));
  }

  // Compute batch norm (inference mode)
  std::vector<float> result(totalElements);
  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t c = 0; c < C; ++c) {
      float invStd = 1.0f / std::sqrt(varData[c] + eps);
      float scale = weight ? wData[c] * invStd : invStd;
      float shift =
          bias ? bData[c] - meanData[c] * scale : -meanData[c] * scale;

      size_t base = (n * C + c) * spatialSize;
      for (size_t s = 0; s < spatialSize; ++s) {
        result[base + s] = data[base + s] * scale + shift;
      }
    }
  }

  Tensor out = runtime_->createTensorEmpty(shape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
}

// =========================================================================
// Embedding ops
// =========================================================================

Tensor Operations::embedding(const Tensor &indices,
                             const Tensor &weight,
                             std::optional<uint32_t> spec) {
  auto node =
      std::make_unique<EmbeddingOpNode>(*runtime_, indices, weight, spec);
  return recordOrEncode(std::move(node), {indices, weight});
}

// =========================================================================
// Padding ops
// =========================================================================

Tensor Operations::pad(const Tensor &input,
                       const std::vector<uint32_t> &padWidths,
                       float value,
                       std::optional<uint32_t> spec) {
  auto node = std::make_unique<PadOpNode>(
      *runtime_, input, std::vector<uint32_t>(padWidths), value, spec);
  return recordOrEncode(std::move(node), {input});
}

// =========================================================================
// Sort (in-place)
// =========================================================================

void Operations::sortBitonic(const Tensor &keys,
                             const Tensor &vals,
                             std::optional<uint32_t> spec) {
  if (graph_) {
    throw std::runtime_error(
        "sortBitonic: in-place sort not supported in graph mode");
  }
  runtime_->encodeOperator(
      std::make_unique<BitonicSortOpNode>(*runtime_, keys, vals, spec));
}

void Operations::sortRadix(const Tensor &keys,
                           const Tensor &vals,
                           std::optional<uint32_t> spec) {
  if (graph_) {
    throw std::runtime_error(
        "sortRadix: in-place sort not supported in graph mode");
  }
  runtime_->encodeOperator(
      std::make_unique<RadixSortOpNode>(*runtime_, keys, vals, spec));
}

} // namespace cut
