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
 * Works directly on Tensor objects and uses the Runtime for GPU
 * dispatch. Retrieves tensor metadata (shape, dtype) via Runtime::getBuffer().
 */
class Operations {
public:
  explicit Operations(Runtime &runtime);

  // ===== Generic element-wise ops =====

  Tensor binaryOp(OperatorEnum op, const Tensor &a, const Tensor &b);

  Tensor unaryOp(OperatorEnum op, const Tensor &a);

  Tensor vecScalarOp(OperatorEnum op, const Tensor &a, DataReference scalar);

  // ===== Reduction ops =====

  /// Reduction. Without dim: global reduction returning shape {1}.
  /// With dim: dimension-wise reduction (dim is removed from output shape).
  /// Always pass the global op enum (e.g. ReduceSum); the dim variant is
  /// resolved automatically.
  Tensor reduce(OperatorEnum op, const Tensor &a, std::optional<int> dim = {});

  // ===== Matrix ops =====

  Tensor matmul(const Tensor &a,
                const Tensor &b,
                OperatorEnum variant = OperatorEnum::MatMul);
  Tensor transpose(const Tensor &a);
  Tensor dot(const Tensor &a, const Tensor &b);

  // ===== Special ops =====

  Tensor clamp(const Tensor &a, DataReference clampData);

  Tensor where(const Tensor &cond, const Tensor &x, const Tensor &y);

  // ===== Cumulative ops =====

  Tensor cumOp(const Tensor &a, OperatorEnum op, std::optional<int> dim = {});

  // ===== Statistical ops =====

  Tensor variance(const Tensor &a, int correction, std::optional<int> dim = {});

  // ===== Softmax =====

  Tensor softmax(const Tensor &a, int dim);
  Tensor logSoftmax(const Tensor &a, int dim);

  // ===== Tensor creation =====

  Tensor arange(DataReference start,
                DataReference end,
                DataReference step,
                DataType dtype);
  Tensor
  linspace(DataReference start, DataReference end, int steps, DataType dtype);
  Tensor full(const std::vector<uint32_t> &shape,
              DataReference fillValue,
              DataType dtype);

  // ===== Shape ops (copy data to new buffer with new shape) =====

  Tensor reshape(const Tensor &a, const std::vector<int32_t> &newShape);
  Tensor squeeze(const Tensor &a, std::optional<int> dim);
  Tensor unsqueeze(const Tensor &a, int dim);
  Tensor
  unflatten(const Tensor &a, int dim, const std::vector<uint32_t> &sizes);
  Tensor flatten(const Tensor &a, int startDim, int endDim);

  // ===== Norm =====

  Tensor norm(const Tensor &a, std::optional<int> dim = {});

  // ===== Prefix scan =====

  Tensor prefixScan(const Tensor &a, OperatorEnum op);

  // ===== Convolution ops =====

  Tensor conv1d(const Tensor &input,
                const Tensor &weight,
                uint32_t stride = 1,
                uint32_t padding = 0);

  Tensor conv2d(const Tensor &input,
                const Tensor &weight,
                uint32_t strideH = 1,
                uint32_t strideW = 1,
                uint32_t padH = 0,
                uint32_t padW = 0);

  // ===== Pooling ops =====

  Tensor maxPool2d(const Tensor &input,
                   uint32_t kernelH,
                   uint32_t kernelW,
                   uint32_t strideH = 1,
                   uint32_t strideW = 1,
                   uint32_t padH = 0,
                   uint32_t padW = 0);

  Tensor avgPool2d(const Tensor &input,
                   uint32_t kernelH,
                   uint32_t kernelW,
                   uint32_t strideH = 1,
                   uint32_t strideW = 1,
                   uint32_t padH = 0,
                   uint32_t padW = 0);

  Tensor adaptiveAvgPool2d(const Tensor &input, uint32_t outH, uint32_t outW);

  // ===== Normalization ops =====

  Tensor layerNorm(const Tensor &input,
                   const std::vector<uint32_t> &normalizedShape,
                   const Tensor *weight = nullptr,
                   const Tensor *bias = nullptr,
                   float eps = 1e-5f);

  Tensor batchNorm(const Tensor &input,
                   const Tensor &runningMean,
                   const Tensor &runningVar,
                   const Tensor *weight = nullptr,
                   const Tensor *bias = nullptr,
                   float eps = 1e-5f);

  // ===== Embedding ops =====

  Tensor embedding(const Tensor &indices, const Tensor &weight);

  // ===== Padding ops =====

  Tensor pad(const Tensor &input,
             const std::vector<uint32_t> &padWidths,
             float value = 0.0f);

  // ===== Sort (in-place) =====

  void sortBitonic(const Tensor &keys, const Tensor &vals);
  void sortRadix(const Tensor &keys, const Tensor &vals);

private:
  friend class Runtime;
  Runtime *runtime_;

  Tensor createOutput(const std::vector<uint32_t> &shape, DataType dtype);

  /// Returns the unpadded shape for a tensor handle.
  std::vector<uint32_t> getShape(const Tensor &h) const;

  /// Returns the dtype for a tensor handle.
  DataType getDtype(const Tensor &h) const;

  /// Dispatches a Copy shader to copy data from src to dst with different
  /// innermost-dim alignments.
  void encodeCopy(const Tensor &src,
                  const Tensor &dst,
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
