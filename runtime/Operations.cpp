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

ComputeHandle Operations::createOutput(const std::vector<uint32_t> &shape,
                                       DataType dtype) {
  return runtime_->createTensorEmpty(shape, dtype);
}

std::vector<uint32_t> Operations::getShape(const ComputeHandle &h) const {
  return runtime_->getTensor(h).getShape();
}

DataType Operations::getDtype(const ComputeHandle &h) const {
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

void Operations::encodeCopy(const ComputeHandle &src,
                            const ComputeHandle &dst,
                            const std::vector<uint32_t> &srcShape,
                            const std::vector<uint32_t> &dstShape) {
  // Compute innermost dim and alignment for source
  uint32_t srcInner = srcShape.empty() ? 1 : srcShape.back();
  uint32_t srcAlignedInner = (srcInner + 3) & ~static_cast<uint32_t>(3);

  // Compute innermost dim and alignment for destination
  uint32_t dstInner = dstShape.empty() ? 1 : dstShape.back();
  uint32_t dstAlignedInner = (dstInner + 3) & ~static_cast<uint32_t>(3);

  // Compute actual inner dim and number of rows
  // actualInnerDim is the unpadded innermost dim of the destination
  uint32_t actualInnerDim = dstInner;
  uint32_t numRows = 1;
  for (size_t i = 0; i + 1 < dstShape.size(); ++i)
    numRows *= dstShape[i];
  if (dstShape.empty())
    numRows = 1;

  uint32_t layoutData[4] = {srcAlignedInner, dstAlignedInner, actualInnerDim,
                            numRows};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, src);
  bindings.emplace_back(1, dst);
  bindings.emplace_back(2, DataReference(layoutData, sizeof(layoutData)));

  runtime_->encodeOperator(OperatorEnum::Copy, bindings);
}

// =========================================================================
// Generic element-wise ops
// =========================================================================

ComputeHandle Operations::binaryOp(OperatorEnum op,
                                   const ComputeHandle &a,
                                   const ComputeHandle &b) {
  const auto &bufA = runtime_->getTensor(a);
  const auto &bufB = runtime_->getTensor(b);

  if (bufA.calculateActualSize() != bufB.calculateActualSize()) {
    throw std::runtime_error(
        "Size mismatch: " + std::to_string(bufA.calculateActualSize()) +
        " vs " + std::to_string(bufB.calculateActualSize()));
  }

  auto shape = getShape(a);
  auto dtype = getDtype(a);
  ComputeHandle output = createOutput(shape, dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, b);
  bindings.emplace_back(2, output);

  runtime_->encodeOperator(op, bindings);
  return output;
}

ComputeHandle Operations::unaryOp(OperatorEnum op, const ComputeHandle &a) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  ComputeHandle output = createOutput(shape, dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, output);

  runtime_->encodeOperator(op, bindings);
  return output;
}

ComputeHandle
Operations::vecScalarOp(OperatorEnum op, const ComputeHandle &a, float scalar) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  ComputeHandle output = createOutput(shape, dtype);

  // Create scalar binding based on dtype
  ComputeBinding scalarBinding = [&]() -> ComputeBinding {
    if (dtype == DataType::Int32) {
      int32_t val = static_cast<int32_t>(scalar);
      return ComputeBinding(2, DataReference(&val, sizeof(val)));
    } else if (dtype == DataType::UInt32) {
      uint32_t val = static_cast<uint32_t>(scalar);
      return ComputeBinding(2, DataReference(&val, sizeof(val)));
    } else {
      return ComputeBinding(2, DataReference(&scalar, sizeof(scalar)));
    }
  }();

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, output);
  bindings.push_back(std::move(scalarBinding));

  runtime_->encodeOperator(op, bindings);
  return output;
}

// =========================================================================
// Reduction ops
// =========================================================================

float Operations::reduceScalar(OperatorEnum op, const ComputeHandle &a) {
  ComputeHandle out = createOutput({1}, DataType::Float32);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, out);

  runtime_->encodeOperator(op, bindings);

  float result = 0.0f;
  runtime_->copyFromTensor(out, &result, sizeof(float));
  return result;
}

bool Operations::reduceBool(OperatorEnum op, const ComputeHandle &a) {
  float val = reduceScalar(op, a);
  return val != 0.0f;
}

int Operations::reduceInt(OperatorEnum op, const ComputeHandle &a) {
  float val = reduceScalar(op, a);
  return static_cast<int>(val);
}

ComputeHandle
Operations::reduceDim(const ComputeHandle &a, int dim, OperatorEnum dimOp) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  auto params = computeDimParams(shape, dim);

  ComputeHandle out = createOutput(params.outShape, dtype);

  uint32_t shapeData[3] = {params.outerSize, params.reduceSize,
                           params.innerSize};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, out);
  bindings.emplace_back(2, DataReference(shapeData, sizeof(shapeData)));

  runtime_->encodeOperator(dimOp, bindings);
  return out;
}

// =========================================================================
// Matrix ops
// =========================================================================

ComputeHandle Operations::matmul(const ComputeHandle &a,
                                 const ComputeHandle &b) {
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

  ComputeHandle output = createOutput({M, N}, DataType::Float32);

  uint32_t shapeData[3] = {M, K, N};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, b);
  bindings.emplace_back(2, output);
  bindings.emplace_back(3, DataReference(shapeData, sizeof(shapeData)));

  runtime_->encodeOperator(OperatorEnum::MatMul, bindings);
  return output;
}

ComputeHandle Operations::transpose(const ComputeHandle &a) {
  auto shape = getShape(a);

  if (shape.size() != 2) {
    throw std::runtime_error("transpose requires a 2D matrix");
  }

  uint32_t M = shape[0], N = shape[1];

  ComputeHandle output = createOutput({N, M}, DataType::Float32);

  uint32_t shapeData[2] = {M, N};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, output);
  bindings.emplace_back(2, DataReference(shapeData, sizeof(shapeData)));

  runtime_->encodeOperator(OperatorEnum::Transpose, bindings);
  return output;
}

float Operations::dot(const ComputeHandle &a, const ComputeHandle &b) {
  const auto &bufA = runtime_->getTensor(a);
  const auto &bufB = runtime_->getTensor(b);

  if (bufA.calculateActualSize() != bufB.calculateActualSize()) {
    throw std::runtime_error(
        "Vector size mismatch: " + std::to_string(bufA.calculateActualSize()) +
        " vs " + std::to_string(bufB.calculateActualSize()));
  }

  auto shape = getShape(a);
  uint32_t count = static_cast<uint32_t>(shapeProduct(shape));

  ComputeHandle out = createOutput({1}, DataType::Float32);

  uint32_t countData[1] = {count};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, b);
  bindings.emplace_back(2, out);
  bindings.emplace_back(3, DataReference(countData, sizeof(countData)));

  runtime_->encodeOperator(OperatorEnum::Dot, bindings);

  float result = 0.0f;
  runtime_->copyFromTensor(out, &result, sizeof(float));
  return result;
}

// =========================================================================
// Special ops
// =========================================================================

ComputeHandle
Operations::clamp(const ComputeHandle &a, float minVal, float maxVal) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  ComputeHandle output = createOutput(shape, dtype);

  // Pack min/max as typed data
  uint8_t clampBuf[8];
  uint32_t clampSize = 0;
  if (dtype == DataType::Int32) {
    int32_t vals[2] = {static_cast<int32_t>(minVal),
                       static_cast<int32_t>(maxVal)};
    std::memcpy(clampBuf, vals, sizeof(vals));
    clampSize = sizeof(vals);
  } else if (dtype == DataType::UInt32) {
    uint32_t vals[2] = {static_cast<uint32_t>(minVal),
                        static_cast<uint32_t>(maxVal)};
    std::memcpy(clampBuf, vals, sizeof(vals));
    clampSize = sizeof(vals);
  } else {
    float vals[2] = {minVal, maxVal};
    std::memcpy(clampBuf, vals, sizeof(vals));
    clampSize = sizeof(vals);
  }

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, output);
  bindings.emplace_back(2, DataReference(clampBuf, clampSize));

  runtime_->encodeOperator(OperatorEnum::TernaryClamp, bindings);
  return output;
}

ComputeHandle Operations::where(const ComputeHandle &cond,
                                const ComputeHandle &x,
                                const ComputeHandle &y) {
  const auto &bufCond = runtime_->getTensor(cond);
  const auto &bufX = runtime_->getTensor(x);
  const auto &bufY = runtime_->getTensor(y);

  if (bufCond.calculateActualSize() != bufX.calculateActualSize() ||
      bufCond.calculateActualSize() != bufY.calculateActualSize()) {
    throw std::runtime_error("condition, x, and y must have the same size");
  }

  auto shape = getShape(x);
  auto dtype = getDtype(x);
  ComputeHandle output = createOutput(shape, dtype);

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

ComputeHandle
Operations::cumOp(const ComputeHandle &a, int dim, OperatorEnum op) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  auto params = computeDimParams(shape, dim);

  ComputeHandle out = createOutput(shape, dtype);

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

float Operations::varianceScalar(const ComputeHandle &a, int correction) {
  auto shape = getShape(a);

  // Compute mean on GPU
  float total = reduceScalar(OperatorEnum::ReduceSum, a);
  size_t n = shapeProduct(shape);
  float m = total / static_cast<float>(n);

  // Read data from GPU
  std::vector<float> data(n);
  runtime_->copyFromTensor(a, data.data(), n * sizeof(float));

  // Compute variance on CPU
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double diff = static_cast<double>(data[i]) - static_cast<double>(m);
    sum += diff * diff;
  }

  int denom = static_cast<int>(n) - correction;
  return denom > 0 ? static_cast<float>(sum / denom) : 0.0f;
}

ComputeHandle
Operations::varianceDim(const ComputeHandle &a, int dim, int correction) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  auto params = computeDimParams(shape, dim);

  // Compute mean along dim using GPU
  ComputeHandle meanHandle = reduceDim(a, dim, OperatorEnum::ReduceDimMean);

  // Read input and mean data from GPU
  size_t totalElements = shapeProduct(shape);
  size_t meanElements = params.outerSize * params.innerSize;

  std::vector<float> flatData(totalElements);
  std::vector<float> meanData(meanElements);

  runtime_->copyFromTensor(a, flatData.data(), totalElements * sizeof(float));
  runtime_->copyFromTensor(meanHandle, meanData.data(),
                           meanElements * sizeof(float));

  // Compute variance on CPU
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

  // Create output tensor and copy result
  ComputeHandle out = createOutput(params.outShape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
}

// =========================================================================
// Softmax
// =========================================================================

ComputeHandle Operations::softmax(const ComputeHandle &a, int dim) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  int ndim = static_cast<int>(shape.size());
  if (dim < 0)
    dim = ndim + dim;

  auto params = computeDimParams(shape, dim);

  // Compute dim-wise max for numerical stability (on GPU)
  ComputeHandle maxHandle = reduceDim(a, dim, OperatorEnum::ReduceDimMax);

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

  ComputeHandle out = createOutput(shape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
}

ComputeHandle Operations::logSoftmax(const ComputeHandle &a, int dim) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  int ndim = static_cast<int>(shape.size());
  if (dim < 0)
    dim = ndim + dim;

  auto params = computeDimParams(shape, dim);

  // Compute dim-wise max for numerical stability (on GPU)
  ComputeHandle maxHandle = reduceDim(a, dim, OperatorEnum::ReduceDimMax);

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

  ComputeHandle out = createOutput(shape, dtype);
  runtime_->copyToTensor(out, result.data(), result.size() * sizeof(float));
  return out;
}

// =========================================================================
// Tensor creation
// =========================================================================

ComputeHandle
Operations::arange(float start, float end, float step, DataType dtype) {
  if (step == 0.0f) {
    throw std::runtime_error("step cannot be zero");
  }

  int n = static_cast<int>((end - start) / step);
  if (n < 0)
    n = 0;

  std::vector<uint32_t> shape = {static_cast<uint32_t>(n)};

  if (dtype == DataType::Int32) {
    std::vector<int32_t> values(n);
    for (int i = 0; i < n; ++i)
      values[i] = static_cast<int32_t>(start + i * step);
    return runtime_->createTensor(shape, dtype, values.data());
  } else if (dtype == DataType::UInt32) {
    std::vector<uint32_t> values(n);
    for (int i = 0; i < n; ++i)
      values[i] = static_cast<uint32_t>(start + i * step);
    return runtime_->createTensor(shape, dtype, values.data());
  } else {
    std::vector<float> values(n);
    for (int i = 0; i < n; ++i)
      values[i] = start + i * step;
    return runtime_->createTensor(shape, dtype, values.data());
  }
}

ComputeHandle
Operations::linspace(float start, float end, int steps, DataType dtype) {
  if (steps < 1) {
    throw std::runtime_error("steps must be at least 1");
  }

  std::vector<float> values(steps);
  if (steps == 1) {
    values[0] = start;
  } else {
    float stepSize = (end - start) / (steps - 1);
    for (int i = 0; i < steps; ++i)
      values[i] = start + i * stepSize;
  }

  std::vector<uint32_t> shape = {static_cast<uint32_t>(steps)};
  return runtime_->createTensor(shape, dtype, values.data());
}

ComputeHandle Operations::full(const std::vector<uint32_t> &shape,
                               float fillValue,
                               DataType dtype) {
  size_t totalSize = shapeProduct(shape);

  if (dtype == DataType::Int32) {
    std::vector<int32_t> values(totalSize, static_cast<int32_t>(fillValue));
    return runtime_->createTensor(shape, dtype, values.data());
  } else if (dtype == DataType::UInt32) {
    std::vector<uint32_t> values(totalSize, static_cast<uint32_t>(fillValue));
    return runtime_->createTensor(shape, dtype, values.data());
  } else {
    std::vector<float> values(totalSize, fillValue);
    return runtime_->createTensor(shape, dtype, values.data());
  }
}

// =========================================================================
// Shape ops (copy data to new buffer with new shape)
// =========================================================================

ComputeHandle Operations::reshape(const ComputeHandle &a,
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

  ComputeHandle output = createOutput(resolved, dtype);
  encodeCopy(a, output, oldShape, resolved);
  return output;
}

ComputeHandle Operations::squeeze(const ComputeHandle &a,
                                  std::optional<int> dim) {
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

  ComputeHandle output = createOutput(newShape, dtype);
  encodeCopy(a, output, oldShape, newShape);
  return output;
}

ComputeHandle Operations::unsqueeze(const ComputeHandle &a, int dim) {
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

  ComputeHandle output = createOutput(newShape, dtype);
  encodeCopy(a, output, oldShape, newShape);
  return output;
}

ComputeHandle Operations::unflatten(const ComputeHandle &a,
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

  ComputeHandle output = createOutput(newShape, dtype);
  encodeCopy(a, output, oldShape, newShape);
  return output;
}

ComputeHandle
Operations::flatten(const ComputeHandle &a, int startDim, int endDim) {
  auto oldShape = getShape(a);
  auto dtype = getDtype(a);
  int ndim = static_cast<int>(oldShape.size());
  if (ndim == 0) {
    // Return a copy for empty shape
    ComputeHandle output = createOutput(oldShape, dtype);
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

  ComputeHandle output = createOutput(newShape, dtype);
  encodeCopy(a, output, oldShape, newShape);
  return output;
}

// =========================================================================
// Norm
// =========================================================================

ComputeHandle Operations::normDim(const ComputeHandle &a, int dim) {
  return reduceDim(a, dim, OperatorEnum::NormDim);
}

// =========================================================================
// Prefix scan
// =========================================================================

ComputeHandle Operations::prefixScan(const ComputeHandle &a, OperatorEnum op) {
  auto shape = getShape(a);
  auto dtype = getDtype(a);
  ComputeHandle out = createOutput(shape, dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a);
  bindings.emplace_back(1, out);

  runtime_->encodeOperator(op, bindings);
  return out;
}

// =========================================================================
// Sort (in-place)
// =========================================================================

void Operations::sortBitonic(const ComputeHandle &keys,
                             const ComputeHandle &vals) {
  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, keys);
  bindings.emplace_back(1, vals);

  runtime_->encodeOperator(OperatorEnum::SortBitonic, bindings);
}

void Operations::sortRadix(const ComputeHandle &keys,
                           const ComputeHandle &vals) {
  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, keys);
  bindings.emplace_back(1, vals);

  runtime_->encodeOperator(OperatorEnum::SortRadix, bindings);
}

} // namespace cut
