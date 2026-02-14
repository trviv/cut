#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeOps.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace cut {

class Runtime;
struct ComputeBuffer;

/**
 * High-level tensor operations implemented in C++.
 * Works directly on ComputeHandle objects and uses the Runtime for GPU
 * dispatch. Retrieves tensor metadata (shape, dtype) via Runtime::getBuffer().
 */
class Operations {
public:
  explicit Operations(Runtime &runtime);

  // ===== Generic element-wise ops =====

  ComputeHandle
  binaryOp(OperatorEnum op, const ComputeHandle &a, const ComputeHandle &b);

  ComputeHandle unaryOp(OperatorEnum op, const ComputeHandle &a);

  ComputeHandle
  vecScalarOp(OperatorEnum op, const ComputeHandle &a, float scalar);

  // ===== Reduction ops =====

  /// Global reduction returning a float scalar.
  float reduceScalar(OperatorEnum op, const ComputeHandle &a);

  /// Global reduction returning a bool (for any/all).
  bool reduceBool(OperatorEnum op, const ComputeHandle &a);

  /// Global reduction returning an int (for argmax/argmin).
  int reduceInt(OperatorEnum op, const ComputeHandle &a);

  /// Dimension-wise reduction.
  ComputeHandle reduceDim(const ComputeHandle &a, int dim, OperatorEnum dimOp);

  // ===== Matrix ops =====

  ComputeHandle matmul(const ComputeHandle &a, const ComputeHandle &b);
  ComputeHandle transpose(const ComputeHandle &a);
  float dot(const ComputeHandle &a, const ComputeHandle &b);

  // ===== Special ops =====

  ComputeHandle clamp(const ComputeHandle &a, float minVal, float maxVal);

  ComputeHandle where(const ComputeHandle &cond,
                      const ComputeHandle &x,
                      const ComputeHandle &y);

  // ===== Cumulative ops =====

  ComputeHandle cumOp(const ComputeHandle &a, int dim, OperatorEnum op);

  // ===== Statistical ops =====

  float varianceScalar(const ComputeHandle &a, int correction);
  ComputeHandle varianceDim(const ComputeHandle &a, int dim, int correction);

  // ===== Softmax =====

  ComputeHandle softmax(const ComputeHandle &a, int dim);
  ComputeHandle logSoftmax(const ComputeHandle &a, int dim);

  // ===== Tensor creation =====

  ComputeHandle arange(float start, float end, float step, DataType dtype);
  ComputeHandle linspace(float start, float end, int steps, DataType dtype);
  ComputeHandle
  full(const std::vector<uint32_t> &shape, float fillValue, DataType dtype);

  // ===== Shape ops (copy data to new buffer with new shape) =====

  ComputeHandle reshape(const ComputeHandle &a,
                        const std::vector<int32_t> &newShape);
  ComputeHandle squeeze(const ComputeHandle &a, std::optional<int> dim);
  ComputeHandle unsqueeze(const ComputeHandle &a, int dim);
  ComputeHandle unflatten(const ComputeHandle &a,
                          int dim,
                          const std::vector<uint32_t> &sizes);
  ComputeHandle flatten(const ComputeHandle &a, int startDim, int endDim);

  // ===== Norm =====

  ComputeHandle normDim(const ComputeHandle &a, int dim);

private:
  Runtime &runtime_;

  ComputeHandle createOutput(const std::vector<uint32_t> &shape,
                             DataType dtype);

  /// Returns the unpadded shape for a tensor handle.
  std::vector<uint32_t> getShape(const ComputeHandle &h) const;

  /// Returns the dtype for a tensor handle.
  DataType getDtype(const ComputeHandle &h) const;

  /// Dispatches a Copy shader to copy data from src to dst with different
  /// innermost-dim alignments.
  void encodeCopy(const ComputeHandle &src,
                  const ComputeHandle &dst,
                  const std::vector<uint32_t> &srcShape,
                  const std::vector<uint32_t> &dstShape);

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
