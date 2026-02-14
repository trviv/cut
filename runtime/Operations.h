#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeOps.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace cut {

class Runtime;

/**
 * Lightweight view of a tensor: handle + dtype + shape.
 * Does not own the underlying GPU buffer.
 */
struct TensorView {
  ComputeHandle handle;
  DataType dtype = DataType::Float32;
  std::vector<uint32_t> shape;

  /// Total number of elements.
  size_t numElements() const;
  /// Total size in bytes.
  size_t sizeBytes() const;
};

/**
 * High-level tensor operations implemented in C++.
 * Replaces the Python operation implementations in compute.py.
 * Works on TensorView objects and uses the Runtime for GPU dispatch.
 */
class Operations {
public:
  explicit Operations(Runtime &runtime);

  // ===== Generic element-wise ops =====

  TensorView
  binaryOp(OperatorEnum op, const TensorView &a, const TensorView &b);

  TensorView unaryOp(OperatorEnum op, const TensorView &a);

  TensorView vecScalarOp(OperatorEnum op, const TensorView &a, float scalar);

  // ===== Reduction ops =====

  /// Global reduction returning a float scalar.
  float reduceScalar(OperatorEnum op, const TensorView &a);

  /// Global reduction returning a bool (for any/all).
  bool reduceBool(OperatorEnum op, const TensorView &a);

  /// Global reduction returning an int (for argmax/argmin).
  int reduceInt(OperatorEnum op, const TensorView &a);

  /// Dimension-wise reduction.
  TensorView reduceDim(const TensorView &a, int dim, OperatorEnum dimOp);

  // ===== Matrix ops =====

  TensorView matmul(const TensorView &a, const TensorView &b);
  TensorView transpose(const TensorView &a);
  float dot(const TensorView &a, const TensorView &b);

  // ===== Special ops =====

  TensorView clamp(const TensorView &a, float minVal, float maxVal);

  TensorView
  where(const TensorView &cond, const TensorView &x, const TensorView &y);

  // ===== Cumulative ops =====

  TensorView cumOp(const TensorView &a, int dim, OperatorEnum op);

  // ===== Statistical ops =====

  float varianceScalar(const TensorView &a, int correction);
  TensorView varianceDim(const TensorView &a, int dim, int correction);

  // ===== Softmax =====

  TensorView softmax(const TensorView &a, int dim);
  TensorView logSoftmax(const TensorView &a, int dim);

  // ===== Tensor creation =====

  TensorView arange(float start, float end, float step, DataType dtype);
  TensorView linspace(float start, float end, int steps, DataType dtype);
  TensorView
  full(const std::vector<uint32_t> &shape, float fillValue, DataType dtype);

  // ===== Shape ops (return views - same handle, new shape) =====

  TensorView reshape(const TensorView &a, const std::vector<int32_t> &newShape);
  TensorView squeeze(const TensorView &a, std::optional<int> dim);
  TensorView unsqueeze(const TensorView &a, int dim);
  TensorView
  unflatten(const TensorView &a, int dim, const std::vector<uint32_t> &sizes);
  TensorView flatten(const TensorView &a, int startDim, int endDim);

  // ===== Norm =====

  TensorView normDim(const TensorView &a, int dim);

private:
  Runtime &runtime_;

  TensorView createOutput(const std::vector<uint32_t> &shape, DataType dtype);

  struct DimParams {
    uint32_t outerSize;
    uint32_t reduceSize;
    uint32_t innerSize;
    std::vector<uint32_t> outShape;
  };

  DimParams computeDimParams(const std::vector<uint32_t> &shape, int dim);

  size_t shapeProduct(const std::vector<uint32_t> &shape) const;
};

} // namespace cut
