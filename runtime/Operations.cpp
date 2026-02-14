#include "Operations.h"
#include "Runtime.h"

#include <ComputeStructs.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cut {

// =========================================================================
// TensorView
// =========================================================================

size_t TensorView::numElements() const {
  if (shape.empty())
    return 0;
  size_t prod = 1;
  for (uint32_t dim : shape)
    prod *= dim;
  return prod;
}

size_t TensorView::sizeBytes() const {
  return numElements() * dataTypeSize(dtype);
}

// =========================================================================
// Operations
// =========================================================================

Operations::Operations(Runtime &runtime) : runtime_(runtime) {}

TensorView Operations::createOutput(const std::vector<uint32_t> &shape,
                                    DataType dtype) {
  TensorView out;
  out.handle = runtime_.createTensorEmpty(shape, dtype);
  out.dtype = dtype;
  out.shape = shape;
  return out;
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

// =========================================================================
// Generic element-wise ops
// =========================================================================

TensorView Operations::binaryOp(OperatorEnum op,
                                const TensorView &a,
                                const TensorView &b) {
  if (a.sizeBytes() != b.sizeBytes()) {
    throw std::runtime_error("Size mismatch: " + std::to_string(a.sizeBytes()) +
                             " vs " + std::to_string(b.sizeBytes()));
  }

  TensorView output = createOutput(a.shape, a.dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a.handle);
  bindings.emplace_back(1, b.handle);
  bindings.emplace_back(2, output.handle);

  runtime_.encodeOperator(op, bindings);
  return output;
}

TensorView Operations::unaryOp(OperatorEnum op, const TensorView &a) {
  TensorView output = createOutput(a.shape, a.dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a.handle);
  bindings.emplace_back(1, output.handle);

  runtime_.encodeOperator(op, bindings);
  return output;
}

TensorView
Operations::vecScalarOp(OperatorEnum op, const TensorView &a, float scalar) {
  TensorView output = createOutput(a.shape, a.dtype);

  // Create scalar binding based on dtype
  ComputeBinding scalarBinding = [&]() -> ComputeBinding {
    if (a.dtype == DataType::Int32) {
      int32_t val = static_cast<int32_t>(scalar);
      return ComputeBinding(2, DataReference(&val, sizeof(val)));
    } else if (a.dtype == DataType::UInt32) {
      uint32_t val = static_cast<uint32_t>(scalar);
      return ComputeBinding(2, DataReference(&val, sizeof(val)));
    } else {
      return ComputeBinding(2, DataReference(&scalar, sizeof(scalar)));
    }
  }();

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a.handle);
  bindings.emplace_back(1, output.handle);
  bindings.push_back(std::move(scalarBinding));

  runtime_.encodeOperator(op, bindings);
  return output;
}

// =========================================================================
// Reduction ops
// =========================================================================

float Operations::reduceScalar(OperatorEnum op, const TensorView &a) {
  TensorView out = createOutput({1}, DataType::Float32);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a.handle);
  bindings.emplace_back(1, out.handle);

  runtime_.encodeOperator(op, bindings);

  float result = 0.0f;
  runtime_.copyFromTensor(out.handle, &result, sizeof(float));
  return result;
}

bool Operations::reduceBool(OperatorEnum op, const TensorView &a) {
  float val = reduceScalar(op, a);
  return val != 0.0f;
}

int Operations::reduceInt(OperatorEnum op, const TensorView &a) {
  float val = reduceScalar(op, a);
  return static_cast<int>(val);
}

TensorView
Operations::reduceDim(const TensorView &a, int dim, OperatorEnum dimOp) {
  auto params = computeDimParams(a.shape, dim);

  TensorView out = createOutput(params.outShape, a.dtype);

  uint32_t shapeData[3] = {params.outerSize, params.reduceSize,
                           params.innerSize};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a.handle);
  bindings.emplace_back(1, out.handle);
  bindings.emplace_back(2, DataReference(shapeData, sizeof(shapeData)));

  runtime_.encodeOperator(dimOp, bindings);
  return out;
}

// =========================================================================
// Matrix ops
// =========================================================================

TensorView Operations::matmul(const TensorView &a, const TensorView &b) {
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw std::runtime_error("matmul requires 2D matrices");
  }

  uint32_t M = a.shape[0], K = a.shape[1];
  uint32_t K2 = b.shape[0], N = b.shape[1];

  if (K != K2) {
    throw std::runtime_error("Matrix dimension mismatch: A is " +
                             std::to_string(M) + "x" + std::to_string(K) +
                             ", B is " + std::to_string(K2) + "x" +
                             std::to_string(N));
  }

  TensorView output = createOutput({M, N}, DataType::Float32);

  uint32_t shapeData[3] = {M, K, N};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a.handle);
  bindings.emplace_back(1, b.handle);
  bindings.emplace_back(2, output.handle);
  bindings.emplace_back(3, DataReference(shapeData, sizeof(shapeData)));

  runtime_.encodeOperator(OperatorEnum::MatMul, bindings);
  return output;
}

TensorView Operations::transpose(const TensorView &a) {
  if (a.shape.size() != 2) {
    throw std::runtime_error("transpose requires a 2D matrix");
  }

  uint32_t M = a.shape[0], N = a.shape[1];

  TensorView output = createOutput({N, M}, DataType::Float32);

  uint32_t shapeData[2] = {M, N};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a.handle);
  bindings.emplace_back(1, output.handle);
  bindings.emplace_back(2, DataReference(shapeData, sizeof(shapeData)));

  runtime_.encodeOperator(OperatorEnum::Transpose, bindings);
  return output;
}

float Operations::dot(const TensorView &a, const TensorView &b) {
  if (a.sizeBytes() != b.sizeBytes()) {
    throw std::runtime_error(
        "Vector size mismatch: " + std::to_string(a.sizeBytes()) + " vs " +
        std::to_string(b.sizeBytes()));
  }

  uint32_t count = static_cast<uint32_t>(shapeProduct(a.shape));

  TensorView out = createOutput({1}, DataType::Float32);

  uint32_t countData[1] = {count};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a.handle);
  bindings.emplace_back(1, b.handle);
  bindings.emplace_back(2, out.handle);
  bindings.emplace_back(3, DataReference(countData, sizeof(countData)));

  runtime_.encodeOperator(OperatorEnum::Dot, bindings);

  float result = 0.0f;
  runtime_.copyFromTensor(out.handle, &result, sizeof(float));
  return result;
}

// =========================================================================
// Special ops
// =========================================================================

TensorView Operations::clamp(const TensorView &a, float minVal, float maxVal) {
  TensorView output = createOutput(a.shape, a.dtype);

  // Pack min/max as typed data
  uint8_t clampBuf[8];
  uint32_t clampSize = 0;
  if (a.dtype == DataType::Int32) {
    int32_t vals[2] = {static_cast<int32_t>(minVal),
                       static_cast<int32_t>(maxVal)};
    std::memcpy(clampBuf, vals, sizeof(vals));
    clampSize = sizeof(vals);
  } else if (a.dtype == DataType::UInt32) {
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
  bindings.emplace_back(0, a.handle);
  bindings.emplace_back(1, output.handle);
  bindings.emplace_back(2, DataReference(clampBuf, clampSize));

  runtime_.encodeOperator(OperatorEnum::TernaryClamp, bindings);
  return output;
}

TensorView Operations::where(const TensorView &cond,
                             const TensorView &x,
                             const TensorView &y) {
  if (cond.sizeBytes() != x.sizeBytes() || cond.sizeBytes() != y.sizeBytes()) {
    throw std::runtime_error("condition, x, and y must have the same size");
  }

  TensorView output = createOutput(x.shape, x.dtype);

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, cond.handle);
  bindings.emplace_back(1, x.handle);
  bindings.emplace_back(2, y.handle);
  bindings.emplace_back(3, output.handle);

  runtime_.encodeOperator(OperatorEnum::TernarySelect, bindings);
  return output;
}

// =========================================================================
// Cumulative ops
// =========================================================================

TensorView Operations::cumOp(const TensorView &a, int dim, OperatorEnum op) {
  auto params = computeDimParams(a.shape, dim);

  TensorView out = createOutput(a.shape, a.dtype);

  uint32_t shapeData[3] = {params.outerSize, params.reduceSize,
                           params.innerSize};

  std::vector<ComputeBinding> bindings;
  bindings.emplace_back(0, a.handle);
  bindings.emplace_back(1, out.handle);
  bindings.emplace_back(2, DataReference(shapeData, sizeof(shapeData)));

  runtime_.encodeOperator(op, bindings);
  return out;
}

// =========================================================================
// Statistical ops
// =========================================================================

float Operations::varianceScalar(const TensorView &a, int correction) {
  // Compute mean on GPU
  float total = reduceScalar(OperatorEnum::ReduceSum, a);
  size_t n = shapeProduct(a.shape);
  float m = total / static_cast<float>(n);

  // Read data from GPU
  std::vector<float> data(n);
  runtime_.copyFromTensor(a.handle, data.data(), n * sizeof(float));

  // Compute variance on CPU
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double diff = static_cast<double>(data[i]) - static_cast<double>(m);
    sum += diff * diff;
  }

  int denom = static_cast<int>(n) - correction;
  return denom > 0 ? static_cast<float>(sum / denom) : 0.0f;
}

TensorView
Operations::varianceDim(const TensorView &a, int dim, int correction) {
  auto params = computeDimParams(a.shape, dim);

  // Compute mean along dim using GPU
  TensorView meanView = reduceDim(a, dim, OperatorEnum::ReduceDimMean);

  // Read input and mean data from GPU
  size_t totalElements = shapeProduct(a.shape);
  size_t meanElements = params.outerSize * params.innerSize;

  std::vector<float> flatData(totalElements);
  std::vector<float> meanData(meanElements);

  runtime_.copyFromTensor(a.handle, flatData.data(),
                          totalElements * sizeof(float));
  runtime_.copyFromTensor(meanView.handle, meanData.data(),
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
  TensorView out = createOutput(params.outShape, a.dtype);
  runtime_.copyToTensor(out.handle, result.data(),
                        result.size() * sizeof(float));
  return out;
}

// =========================================================================
// Softmax
// =========================================================================

TensorView Operations::softmax(const TensorView &a, int dim) {
  int ndim = static_cast<int>(a.shape.size());
  if (dim < 0)
    dim = ndim + dim;

  auto params = computeDimParams(a.shape, dim);

  // Compute dim-wise max for numerical stability (on GPU)
  TensorView maxVals = reduceDim(a, dim, OperatorEnum::ReduceDimMax);

  // Read data from GPU
  size_t totalElements = shapeProduct(a.shape);
  size_t maxElements = params.outerSize * params.innerSize;

  std::vector<float> flatData(totalElements);
  std::vector<float> maxData(maxElements);

  runtime_.copyFromTensor(a.handle, flatData.data(),
                          totalElements * sizeof(float));
  runtime_.copyFromTensor(maxVals.handle, maxData.data(),
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

  TensorView out = createOutput(a.shape, a.dtype);
  runtime_.copyToTensor(out.handle, result.data(),
                        result.size() * sizeof(float));
  return out;
}

TensorView Operations::logSoftmax(const TensorView &a, int dim) {
  int ndim = static_cast<int>(a.shape.size());
  if (dim < 0)
    dim = ndim + dim;

  auto params = computeDimParams(a.shape, dim);

  // Compute dim-wise max for numerical stability (on GPU)
  TensorView maxVals = reduceDim(a, dim, OperatorEnum::ReduceDimMax);

  // Read data from GPU
  size_t totalElements = shapeProduct(a.shape);
  size_t maxElements = params.outerSize * params.innerSize;

  std::vector<float> flatData(totalElements);
  std::vector<float> maxData(maxElements);

  runtime_.copyFromTensor(a.handle, flatData.data(),
                          totalElements * sizeof(float));
  runtime_.copyFromTensor(maxVals.handle, maxData.data(),
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

  TensorView out = createOutput(a.shape, a.dtype);
  runtime_.copyToTensor(out.handle, result.data(),
                        result.size() * sizeof(float));
  return out;
}

// =========================================================================
// Tensor creation
// =========================================================================

TensorView
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
    return TensorView{runtime_.createTensor(shape, dtype, values.data()), dtype,
                      shape};
  } else if (dtype == DataType::UInt32) {
    std::vector<uint32_t> values(n);
    for (int i = 0; i < n; ++i)
      values[i] = static_cast<uint32_t>(start + i * step);
    return TensorView{runtime_.createTensor(shape, dtype, values.data()), dtype,
                      shape};
  } else {
    std::vector<float> values(n);
    for (int i = 0; i < n; ++i)
      values[i] = start + i * step;
    return TensorView{runtime_.createTensor(shape, dtype, values.data()), dtype,
                      shape};
  }
}

TensorView
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
  return TensorView{runtime_.createTensor(shape, dtype, values.data()), dtype,
                    shape};
}

TensorView Operations::full(const std::vector<uint32_t> &shape,
                            float fillValue,
                            DataType dtype) {
  size_t totalSize = shapeProduct(shape);

  if (dtype == DataType::Int32) {
    std::vector<int32_t> values(totalSize, static_cast<int32_t>(fillValue));
    return TensorView{runtime_.createTensor(shape, dtype, values.data()), dtype,
                      shape};
  } else if (dtype == DataType::UInt32) {
    std::vector<uint32_t> values(totalSize, static_cast<uint32_t>(fillValue));
    return TensorView{runtime_.createTensor(shape, dtype, values.data()), dtype,
                      shape};
  } else {
    std::vector<float> values(totalSize, fillValue);
    return TensorView{runtime_.createTensor(shape, dtype, values.data()), dtype,
                      shape};
  }
}

// =========================================================================
// Shape ops (views - same handle, new shape)
// =========================================================================

TensorView Operations::reshape(const TensorView &a,
                               const std::vector<int32_t> &newShape) {
  size_t oldTotal = shapeProduct(a.shape);

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

  return TensorView{a.handle, a.dtype, resolved};
}

TensorView Operations::squeeze(const TensorView &a, std::optional<int> dim) {
  std::vector<uint32_t> newShape;
  int ndim = static_cast<int>(a.shape.size());

  if (!dim.has_value()) {
    // Squeeze all size-1 dims
    for (uint32_t s : a.shape) {
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
      if (i == d && a.shape[i] == 1)
        continue;
      newShape.push_back(a.shape[i]);
    }
  }

  if (newShape.empty())
    newShape.push_back(1);

  return TensorView{a.handle, a.dtype, newShape};
}

TensorView Operations::unsqueeze(const TensorView &a, int dim) {
  int ndim = static_cast<int>(a.shape.size());

  if (dim < 0)
    dim = ndim + 1 + dim;
  if (dim < 0 || dim > ndim) {
    throw std::runtime_error(
        "dim " + std::to_string(dim) + " out of range for tensor with " +
        std::to_string(ndim) + " dimensions (valid range: [" +
        std::to_string(-(ndim + 1)) + ", " + std::to_string(ndim) + "])");
  }

  std::vector<uint32_t> newShape(a.shape.begin(), a.shape.begin() + dim);
  newShape.push_back(1);
  newShape.insert(newShape.end(), a.shape.begin() + dim, a.shape.end());

  return TensorView{a.handle, a.dtype, newShape};
}

TensorView Operations::unflatten(const TensorView &a,
                                 int dim,
                                 const std::vector<uint32_t> &sizes) {
  int ndim = static_cast<int>(a.shape.size());
  if (dim < 0)
    dim = ndim + dim;
  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("dim " + std::to_string(dim) +
                                " out of range for tensor with " +
                                std::to_string(ndim) + " dimensions");
  }

  size_t expected = a.shape[dim];
  size_t actual = 1;
  for (uint32_t s : sizes)
    actual *= s;

  if (expected != actual) {
    throw std::invalid_argument("Product of sizes (" + std::to_string(actual) +
                                ") must match dimension " +
                                std::to_string(dim) + " size (" +
                                std::to_string(expected) + ")");
  }

  std::vector<uint32_t> newShape(a.shape.begin(), a.shape.begin() + dim);
  newShape.insert(newShape.end(), sizes.begin(), sizes.end());
  newShape.insert(newShape.end(), a.shape.begin() + dim + 1, a.shape.end());

  return TensorView{a.handle, a.dtype, newShape};
}

TensorView Operations::flatten(const TensorView &a, int startDim, int endDim) {
  int ndim = static_cast<int>(a.shape.size());
  if (ndim == 0)
    return a;

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
    newShape.push_back(static_cast<uint32_t>(shapeProduct(a.shape)));
  } else {
    newShape.insert(newShape.end(), a.shape.begin(),
                    a.shape.begin() + startDim);
    uint32_t flattenedSize = 1;
    for (int i = startDim; i <= endDim; ++i)
      flattenedSize *= a.shape[i];
    newShape.push_back(flattenedSize);
    newShape.insert(newShape.end(), a.shape.begin() + endDim + 1,
                    a.shape.end());
  }

  return TensorView{a.handle, a.dtype, newShape};
}

// =========================================================================
// Norm
// =========================================================================

TensorView Operations::normDim(const TensorView &a, int dim) {
  return reduceDim(a, dim, OperatorEnum::NormDim);
}

} // namespace cut
