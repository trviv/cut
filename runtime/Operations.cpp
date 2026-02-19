#include "Operations.h"
#include "Runtime.h"

#include <ComputeStructs.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cut {

// =========================================================================
// Operations
// =========================================================================

Operations::Operations(Runtime &runtime) : runtime_(&runtime) {}

Tensor Operations::createOutput(const std::vector<uint32_t> &shape,
                                DataType dtype) {
  return runtime_->createTensorEmpty(shape, dtype);
}

std::vector<uint32_t> Operations::getShape(const Tensor &h) const {
  return runtime_->getTensor(h).getShape();
}

DataType Operations::getDtype(const Tensor &h) const {
  return runtime_->getTensor(h).getDtype();
}

size_t Operations::shapeProduct(const std::vector<uint32_t> &shape) const {
  size_t prod = 1;
  for (uint32_t dim : shape)
    prod *= dim;
  return prod;
}

Operations::DimParams
Operations::computeDimParams(const std::vector<uint32_t> &shape, int dim) {
  int ndim = static_cast<int>(shape.size());
  if (dim < 0)
    dim = ndim + dim;
  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("dim " + std::to_string(dim) +
                                " out of range for tensor with " +
                                std::to_string(ndim) + " dimensions");
  }

  DimParams params;
  params.outerSize = 1;
  for (int i = 0; i < dim; ++i)
    params.outerSize *= shape[i];
  params.reduceSize = shape[dim];
  params.innerSize = 1;
  for (int i = dim + 1; i < ndim; ++i)
    params.innerSize *= shape[i];

  for (int i = 0; i < ndim; ++i) {
    if (i != dim)
      params.outShape.push_back(shape[i]);
  }
  if (params.outShape.empty())
    params.outShape.push_back(1);

  return params;
}

void Operations::encodeCopy(const Tensor &src,
                            const Tensor &dst,
                            const std::vector<uint32_t> &srcShape,
                            const std::vector<uint32_t> &dstShape) {
  // Compute innermost dim and alignment for source
  uint32_t srcInner = srcShape.empty() ? 1 : srcShape.back();
  uint32_t srcAlignedInner = (srcInner + 3) & ~static_cast<uint32_t>(3);

  // Compute innermost dim and alignment for destination
  uint32_t dstInner = dstShape.empty() ? 1 : dstShape.back();
  uint32_t dstAlignedInner = (dstInner + 3) & ~static_cast<uint32_t>(3);

  // Compute total logical elements
  uint32_t totalElements = 1;
  for (size_t i = 0; i < dstShape.size(); ++i)
    totalElements *= dstShape[i];
  if (dstShape.empty())
    totalElements = 1;

  uint32_t layoutData[5] = {srcAlignedInner, srcInner, dstAlignedInner,
                            dstInner, totalElements};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, src);
  bindings.emplace_back(1, dst);
  bindings.emplace_back(2, DataReference(layoutData, sizeof(layoutData)));

  runtime_->encodeOperator(OperatorEnum::Copy, bindings);
}

// =========================================================================
// Generic element-wise ops
// =========================================================================

Tensor Operations::binaryOp(OperatorEnum op, const Tensor &a, const Tensor &b) {
  const auto &bufA = runtime_->getTensor(a);
  const auto &bufB = runtime_->getTensor(b);

  if (bufA.calculateActualSize() != bufB.calculateActualSize()) {
    throw std::runtime_error(
        "Size mismatch: " + std::to_string(bufA.calculateActualSize()) +
        " vs " + std::to_string(bufB.calculateActualSize()));
  }

  auto shape = getShape(a);
  auto dtype = getDtype(a);
  Tensor output = createOutput(shape, dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, b);
  bindings.emplace_back(2, output);

  runtime_->encodeOperator(op, bindings);
  return output;
}

Tensor Operations::unaryOp(OperatorEnum op, const Tensor &a) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  Tensor output = createOutput(shape, dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, output);

  runtime_->encodeOperator(op, bindings);
  return output;
}

Tensor Operations::vecScalarOp(OperatorEnum op,
                               const Tensor &a,
                               DataReference scalar) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  Tensor output = createOutput(shape, dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, output);
  bindings.emplace_back(2, scalar);

  runtime_->encodeOperator(op, bindings);
  return output;
}

// =========================================================================
// Reduction ops
// =========================================================================

Tensor
Operations::reduce(OperatorEnum op, const Tensor &a, std::optional<int> dim) {
  auto dtype = getDtype(a);

  if (!dim.has_value()) {
    // Global reduction → shape {1}
    Tensor out = createOutput({1}, dtype);

    std::vector<ComputeBinding> bindings;
    bindings.emplace_back(0, a);
    bindings.emplace_back(1, out);

    runtime_->encodeOperator(op, bindings);
    return out;
  }

  // Dimension-wise reduction
  auto shape = getShape(a);
  auto params = computeDimParams(shape, dim.value());

  Tensor out = createOutput(params.outShape, dtype);

  uint32_t shapeData[3] = {params.outerSize, params.reduceSize,
                           params.innerSize};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, out);
  bindings.emplace_back(2, DataReference(shapeData, sizeof(shapeData)));

  runtime_->encodeOperator(op, bindings);
  return out;
}

// =========================================================================
// Matrix ops
// =========================================================================

Tensor
Operations::matmul(const Tensor &a, const Tensor &b, OperatorEnum variant) {
  auto shapeA = getShape(a);
  auto shapeB = getShape(b);

  if (shapeA.size() != 2 || shapeB.size() != 2) {
    throw std::runtime_error("matmul requires 2D matrices");
  }

  uint32_t M = shapeA[0], K = shapeA[1];
  uint32_t K2 = shapeB[0], N = shapeB[1];

  if (K != K2) {
    throw std::runtime_error("Matrix dimension mismatch: A is " +
                             std::to_string(M) + "x" + std::to_string(K) +
                             ", B is " + std::to_string(K2) + "x" +
                             std::to_string(N));
  }

  Tensor output = createOutput({M, N}, DataType::Float32);

  uint32_t shapeData[3] = {M, K, N};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, b);
  bindings.emplace_back(2, output);
  bindings.emplace_back(3, DataReference(shapeData, sizeof(shapeData)));

  runtime_->encodeOperator(variant, bindings);
  return output;
}

Tensor Operations::transpose(const Tensor &a) {
  auto shape = getShape(a);

  if (shape.size() != 2) {
    throw std::runtime_error("transpose requires a 2D matrix");
  }

  uint32_t M = shape[0], N = shape[1];

  Tensor output = createOutput({N, M}, DataType::Float32);

  uint32_t shapeData[2] = {M, N};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, output);
  bindings.emplace_back(2, DataReference(shapeData, sizeof(shapeData)));

  runtime_->encodeOperator(OperatorEnum::Transpose, bindings);
  return output;
}

Tensor Operations::dot(const Tensor &a, const Tensor &b) {
  const auto &bufA = runtime_->getTensor(a);
  const auto &bufB = runtime_->getTensor(b);

  if (bufA.calculateActualSize() != bufB.calculateActualSize()) {
    throw std::runtime_error(
        "Vector size mismatch: " + std::to_string(bufA.calculateActualSize()) +
        " vs " + std::to_string(bufB.calculateActualSize()));
  }

  auto shape = getShape(a);
  uint32_t count = static_cast<uint32_t>(shapeProduct(shape));

  // Each workgroup of 256 threads produces one partial sum
  constexpr uint32_t kWorkgroupSize = 256;
  uint32_t numWorkgroups = (count + kWorkgroupSize - 1) / kWorkgroupSize;

  Tensor partials = createOutput({numWorkgroups}, DataType::Float32);

  uint32_t countData[1] = {count};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, b);
  bindings.emplace_back(2, partials);
  bindings.emplace_back(3, DataReference(countData, sizeof(countData)));

  runtime_->encodeOperator(OperatorEnum::Dot, bindings);

  // Read back per-workgroup partial sums and accumulate on CPU
  std::vector<float> partialData(numWorkgroups);
  runtime_->copyFromTensor(partials, partialData.data(),
                           numWorkgroups * sizeof(float));
  float result = 0.0f;
  for (uint32_t i = 0; i < numWorkgroups; ++i) {
    result += partialData[i];
  }

  Tensor out = createOutput({1}, DataType::Float32);
  runtime_->copyToTensor(out, &result, sizeof(float));
  return out;
}

// =========================================================================
// Special ops
// =========================================================================

Tensor Operations::clamp(const Tensor &a, DataReference clampData) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  Tensor output = createOutput(shape, dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, output);
  bindings.emplace_back(2, clampData);

  runtime_->encodeOperator(OperatorEnum::TernaryClamp, bindings);
  return output;
}

Tensor Operations::where(const Tensor &cond, const Tensor &x, const Tensor &y) {
  const auto &bufCond = runtime_->getTensor(cond);
  const auto &bufX = runtime_->getTensor(x);
  const auto &bufY = runtime_->getTensor(y);

  if (bufCond.calculateActualSize() != bufX.calculateActualSize() ||
      bufCond.calculateActualSize() != bufY.calculateActualSize()) {
    throw std::runtime_error("condition, x, and y must have the same size");
  }

  auto shape = getShape(x);
  auto dtype = getDtype(x);
  Tensor output = createOutput(shape, dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, cond);
  bindings.emplace_back(1, x);
  bindings.emplace_back(2, y);
  bindings.emplace_back(3, output);

  runtime_->encodeOperator(OperatorEnum::TernarySelect, bindings);
  return output;
}

// =========================================================================
// Cumulative ops
// =========================================================================

Tensor
Operations::cumOp(const Tensor &a, OperatorEnum op, std::optional<int> dim) {
  int d = dim.value_or(0);
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  auto params = computeDimParams(shape, d);

  Tensor out = createOutput(shape, dtype);

  uint32_t shapeData[3] = {params.outerSize, params.reduceSize,
                           params.innerSize};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, out);
  bindings.emplace_back(2, DataReference(shapeData, sizeof(shapeData)));

  runtime_->encodeOperator(op, bindings);
  return out;
}

// =========================================================================
// Statistical ops
// =========================================================================

Tensor
Operations::variance(const Tensor &a, int correction, std::optional<int> dim) {
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

    Tensor out = createOutput({1}, dtype);
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

  Tensor out = createOutput(params.outShape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
}

// =========================================================================
// Softmax
// =========================================================================

Tensor Operations::softmax(const Tensor &a, int dim) {
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

  Tensor out = createOutput(shape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
}

Tensor Operations::logSoftmax(const Tensor &a, int dim) {
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

  Tensor out = createOutput(shape, dtype);
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
  auto oldShape = getShape(a);
  auto dtype = getDtype(a);
  size_t oldTotal = shapeProduct(oldShape);

  std::vector<uint32_t> resolved;
  int negIdx = -1;
  size_t knownTotal = 1;

  for (int i = 0; i < static_cast<int>(newShape.size()); ++i) {
    if (newShape[i] == -1) {
      if (negIdx != -1)
        throw std::invalid_argument("Only one dimension can be -1");
      negIdx = i;
      resolved.push_back(0); // placeholder
    } else if (newShape[i] < 0) {
      throw std::invalid_argument("Invalid shape dimension: " +
                                  std::to_string(newShape[i]));
    } else {
      resolved.push_back(static_cast<uint32_t>(newShape[i]));
      knownTotal *= newShape[i];
    }
  }

  if (negIdx != -1) {
    if (knownTotal == 0)
      throw std::invalid_argument(
          "Cannot infer dimension with other zero-size dimensions");
    size_t inferred = oldTotal / knownTotal;
    if (inferred * knownTotal != oldTotal) {
      throw std::invalid_argument("Shape is invalid for tensor of size " +
                                  std::to_string(oldTotal));
    }
    resolved[negIdx] = static_cast<uint32_t>(inferred);
  }

  size_t newTotal = shapeProduct(resolved);
  if (oldTotal != newTotal) {
    throw std::invalid_argument("Cannot reshape tensor of size " +
                                std::to_string(oldTotal) + " to new shape");
  }

  Tensor output = createOutput(resolved, dtype);
  encodeCopy(a, output, oldShape, resolved);
  return output;
}

Tensor Operations::squeeze(const Tensor &a, std::optional<int> dim) {
  auto oldShape = getShape(a);
  auto dtype = getDtype(a);
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

  Tensor output = createOutput(newShape, dtype);
  encodeCopy(a, output, oldShape, newShape);
  return output;
}

Tensor Operations::unsqueeze(const Tensor &a, int dim) {
  auto oldShape = getShape(a);
  auto dtype = getDtype(a);
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

  Tensor output = createOutput(newShape, dtype);
  encodeCopy(a, output, oldShape, newShape);
  return output;
}

Tensor Operations::unflatten(const Tensor &a,
                             int dim,
                             const std::vector<uint32_t> &sizes) {
  auto oldShape = getShape(a);
  auto dtype = getDtype(a);
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

  Tensor output = createOutput(newShape, dtype);
  encodeCopy(a, output, oldShape, newShape);
  return output;
}

Tensor Operations::flatten(const Tensor &a, int startDim, int endDim) {
  auto oldShape = getShape(a);
  auto dtype = getDtype(a);
  int ndim = static_cast<int>(oldShape.size());
  if (ndim == 0) {
    // Return a copy for empty shape
    Tensor output = createOutput(oldShape, dtype);
    encodeCopy(a, output, oldShape, oldShape);
    return output;
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

  Tensor output = createOutput(newShape, dtype);
  encodeCopy(a, output, oldShape, newShape);
  return output;
}

// =========================================================================
// Norm
// =========================================================================

Tensor Operations::norm(const Tensor &a, std::optional<int> dim) {
  if (!dim.has_value()) {
    auto dtype = getDtype(a);
    Tensor out = createOutput({1}, dtype);

    std::vector<ComputeBinding> bindings;
    bindings.emplace_back(0, a);
    bindings.emplace_back(1, out);

    runtime_->encodeOperator(OperatorEnum::Norm, bindings);
    return out;
  }
  return reduce(OperatorEnum::NormDim, a, dim.value());
}

// =========================================================================
// Prefix scan
// =========================================================================

Tensor Operations::prefixScan(const Tensor &a, OperatorEnum op) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  Tensor out = createOutput(shape, dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, out);

  runtime_->encodeOperator(op, bindings);
  return out;
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
                          uint32_t padding) {
  auto inShape = getShape(input); // [N, C_in, L_in]
  auto wShape = getShape(weight); // [C_out, C_in, kL]
  auto dtype = getDtype(input);

  if (inShape.size() != 3)
    throw std::runtime_error("conv1d: input must be 3D [N, C_in, L_in]");
  if (wShape.size() != 3)
    throw std::runtime_error("conv1d: weight must be 3D [C_out, C_in, kL]");

  uint32_t N = inShape[0], C_in = inShape[1], L_in = inShape[2];
  uint32_t C_out = wShape[0], kL = wShape[2];

  if (wShape[1] != C_in)
    throw std::runtime_error("conv1d: weight C_in dimension mismatch");

  uint32_t L_out = (L_in + 2 * padding - kL) / stride + 1;

  Tensor output = createOutput({N, C_out, L_out}, dtype);

  struct Conv1DParams {
    uint32_t batchSize, C_in, L_in, C_out, kL;
    uint32_t stride, padding;
  } params{N, C_in, L_in, C_out, kL, stride, padding};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, input);
  bindings.emplace_back(1, weight);
  bindings.emplace_back(2, output);
  bindings.emplace_back(3, DataReference(&params, sizeof(params)));

  runtime_->encodeOperator(OperatorEnum::Conv1D, bindings);
  return output;
}

Tensor Operations::conv2d(const Tensor &input,
                          const Tensor &weight,
                          uint32_t strideH,
                          uint32_t strideW,
                          uint32_t padH,
                          uint32_t padW) {
  auto inShape = getShape(input); // [N, C_in, H_in, W_in]
  auto wShape = getShape(weight); // [C_out, C_in, kH, kW]
  auto dtype = getDtype(input);

  if (inShape.size() != 4)
    throw std::runtime_error("conv2d: input must be 4D [N, C_in, H_in, W_in]");
  if (wShape.size() != 4)
    throw std::runtime_error("conv2d: weight must be 4D [C_out, C_in, kH, kW]");

  uint32_t N = inShape[0], C_in = inShape[1], H_in = inShape[2],
           W_in = inShape[3];
  uint32_t C_out = wShape[0], kH = wShape[2], kW = wShape[3];

  if (wShape[1] != C_in)
    throw std::runtime_error("conv2d: weight C_in dimension mismatch");

  uint32_t H_out = (H_in + 2 * padH - kH) / strideH + 1;
  uint32_t W_out = (W_in + 2 * padW - kW) / strideW + 1;

  Tensor output = createOutput({N, C_out, H_out, W_out}, dtype);

  struct Conv2DParams {
    uint32_t batchSize, C_in, H_in, W_in;
    uint32_t C_out, kH, kW;
    uint32_t strideH, strideW;
    uint32_t padH, padW;
  } params{N, C_in, H_in, W_in, C_out, kH, kW, strideH, strideW, padH, padW};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, input);
  bindings.emplace_back(1, weight);
  bindings.emplace_back(2, output);
  bindings.emplace_back(3, DataReference(&params, sizeof(params)));

  runtime_->encodeOperator(OperatorEnum::Conv2D, bindings);
  return output;
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
                             uint32_t padW) {
  auto inShape = getShape(input); // [N, C, H_in, W_in]
  auto dtype = getDtype(input);

  if (inShape.size() != 4)
    throw std::runtime_error("max_pool2d: input must be 4D [N, C, H, W]");

  uint32_t N = inShape[0], C = inShape[1], H_in = inShape[2], W_in = inShape[3];
  uint32_t H_out = (H_in + 2 * padH - kernelH) / strideH + 1;
  uint32_t W_out = (W_in + 2 * padW - kernelW) / strideW + 1;

  Tensor output = createOutput({N, C, H_out, W_out}, dtype);

  struct Pool2DParams {
    uint32_t N, C, H_in, W_in;
    uint32_t kernelH, kernelW;
    uint32_t strideH, strideW;
    uint32_t padH, padW;
  } params{N, C, H_in, W_in, kernelH, kernelW, strideH, strideW, padH, padW};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, input);
  bindings.emplace_back(1, output);
  bindings.emplace_back(2, DataReference(&params, sizeof(params)));

  runtime_->encodeOperator(OperatorEnum::MaxPool2D, bindings);
  return output;
}

Tensor Operations::avgPool2d(const Tensor &input,
                             uint32_t kernelH,
                             uint32_t kernelW,
                             uint32_t strideH,
                             uint32_t strideW,
                             uint32_t padH,
                             uint32_t padW) {
  auto inShape = getShape(input); // [N, C, H_in, W_in]
  auto dtype = getDtype(input);

  if (inShape.size() != 4)
    throw std::runtime_error("avg_pool2d: input must be 4D [N, C, H, W]");

  uint32_t N = inShape[0], C = inShape[1], H_in = inShape[2], W_in = inShape[3];
  uint32_t H_out = (H_in + 2 * padH - kernelH) / strideH + 1;
  uint32_t W_out = (W_in + 2 * padW - kernelW) / strideW + 1;

  Tensor output = createOutput({N, C, H_out, W_out}, dtype);

  struct Pool2DParams {
    uint32_t N, C, H_in, W_in;
    uint32_t kernelH, kernelW;
    uint32_t strideH, strideW;
    uint32_t padH, padW;
  } params{N, C, H_in, W_in, kernelH, kernelW, strideH, strideW, padH, padW};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, input);
  bindings.emplace_back(1, output);
  bindings.emplace_back(2, DataReference(&params, sizeof(params)));

  runtime_->encodeOperator(OperatorEnum::AvgPool2D, bindings);
  return output;
}

Tensor Operations::adaptiveAvgPool2d(const Tensor &input,
                                     uint32_t outH,
                                     uint32_t outW) {
  auto inShape = getShape(input);
  if (inShape.size() != 4)
    throw std::runtime_error(
        "adaptive_avg_pool2d: input must be 4D [N, C, H, W]");

  uint32_t H_in = inShape[2], W_in = inShape[3];

  // Compute kernel and stride to produce desired output size
  // PyTorch formula: stride = floor(input_size / output_size)
  //                  kernel = input_size - (output_size - 1) * stride
  uint32_t strideH = H_in / outH;
  uint32_t strideW = W_in / outW;
  uint32_t kernelH = H_in - (outH - 1) * strideH;
  uint32_t kernelW = W_in - (outW - 1) * strideW;

  return avgPool2d(input, kernelH, kernelW, strideH, strideW, 0, 0);
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

  Tensor out = createOutput(shape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
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

  // Compute batch norm (inference mode): y = (x - mean) / sqrt(var + eps) * w +
  // b
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

  Tensor out = createOutput(shape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
}

// =========================================================================
// Embedding ops
// =========================================================================

Tensor Operations::embedding(const Tensor &indices, const Tensor &weight) {
  auto idxShape = getShape(indices);
  auto wShape = getShape(weight); // [num_embeddings, embedding_dim]

  if (wShape.size() != 2)
    throw std::runtime_error(
        "embedding: weight must be 2D [num_embeddings, embedding_dim]");

  uint32_t embDim = wShape[1];

  // Output shape = indices shape + [embedding_dim]
  std::vector<uint32_t> outShape(idxShape.begin(), idxShape.end());
  outShape.push_back(embDim);

  Tensor output = createOutput(outShape, DataType::Float32);

  size_t numIndices = shapeProduct(idxShape);

  struct EmbeddingParams {
    uint32_t numIndices;
    uint32_t embDim;
  } params{static_cast<uint32_t>(numIndices), embDim};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, indices);
  bindings.emplace_back(1, weight);
  bindings.emplace_back(2, output);
  bindings.emplace_back(3, DataReference(&params, sizeof(params)));

  runtime_->encodeOperator(OperatorEnum::Embedding, bindings);
  return output;
}

// =========================================================================
// Padding ops
// =========================================================================

Tensor Operations::pad(const Tensor &input,
                       const std::vector<uint32_t> &padWidths,
                       float value) {
  auto shape = getShape(input);
  auto dtype = getDtype(input);
  int ndim = static_cast<int>(shape.size());

  // padWidths is in PyTorch order: (left, right) for last dim,
  // then (left, right) for second-to-last, etc.
  if (padWidths.size() % 2 != 0 ||
      static_cast<int>(padWidths.size() / 2) > ndim) {
    throw std::runtime_error("pad: invalid padWidths length");
  }

  int numPaddedDims = static_cast<int>(padWidths.size() / 2);

  // Build output shape
  std::vector<uint32_t> outShape = shape;
  for (int i = 0; i < numPaddedDims; ++i) {
    int dim = ndim - 1 - i;
    outShape[dim] += padWidths[2 * i] + padWidths[2 * i + 1];
  }

  Tensor output = createOutput(outShape, dtype);

  // Build params: input shape (4) + pad widths (up to 8) + fill value
  // Fixed-size struct for push constants
  struct PadParams {
    uint32_t ndim;
    uint32_t inShape[4];
    uint32_t outShape[4];
    uint32_t padBefore[4]; // padding before each dim (from dim 0)
    uint32_t totalElements;
    float fillValue;
  } params{};

  params.ndim = static_cast<uint32_t>(ndim);
  params.totalElements = static_cast<uint32_t>(shapeProduct(outShape));
  std::memcpy(&params.fillValue, &value, sizeof(float));

  for (int i = 0; i < ndim; ++i) {
    params.inShape[i] = shape[i];
    params.outShape[i] = outShape[i];
    params.padBefore[i] = 0;
  }

  // Map PyTorch padWidths (innermost first) to per-dim padBefore
  for (int i = 0; i < numPaddedDims; ++i) {
    int dim = ndim - 1 - i;
    params.padBefore[dim] = padWidths[2 * i];
  }

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, input);
  bindings.emplace_back(1, output);
  bindings.emplace_back(2, DataReference(&params, sizeof(params)));

  runtime_->encodeOperator(OperatorEnum::Pad, bindings);
  return output;
}

// =========================================================================
// Sort (in-place)
// =========================================================================

void Operations::sortBitonic(const Tensor &keys, const Tensor &vals) {
  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, keys);
  bindings.emplace_back(1, vals);

  runtime_->encodeOperator(OperatorEnum::SortBitonic, bindings);
}

void Operations::sortRadix(const Tensor &keys, const Tensor &vals) {
  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, keys);
  bindings.emplace_back(1, vals);

  runtime_->encodeOperator(OperatorEnum::SortRadix, bindings);
}

} // namespace cut
