#pragma once

#include "Graph.h"
#include "ShapeUtils.h"

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeOps.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace cut {

class Operations;
class Runtime;

namespace graph {

/// Builds a computation graph by routing Operations calls through graph mode.
/// Instead of dispatching to the GPU, each method creates an OpNode via
/// Operations and returns the resulting Tensor. After construction, call
/// build() to obtain the Graph for optimization and execution.
class GraphBuilder {
public:
  explicit GraphBuilder(Runtime &runtime);
  ~GraphBuilder();

  /// Move-returns the constructed graph. The builder should not be used after
  /// this.
  Graph build();

  // === Graph inputs ===

  /// Register a pre-existing GPU tensor as a graph input.
  /// Set isConstant=true for model weights (enables optimizer reasoning).
  Tensor input(const Tensor &gpuHandle, bool isConstant = false);

  // === Element-wise ops ===

  Tensor binaryOp(OperatorEnum op, const Tensor &a, const TensorLike &b);
  Tensor unaryOp(OperatorEnum op, const Tensor &a);

  // === Reduction ===

  Tensor reduce(OperatorEnum op, const Tensor &a, std::optional<int> dim = {});

  // === Matrix ops ===

  Tensor matmul(const Tensor &a, const Tensor &b);
  Tensor transpose(const Tensor &a);
  Tensor dot(const Tensor &a, const Tensor &b);

  // === Special ops ===

  Tensor clamp(const Tensor &a, float minVal, float maxVal);
  Tensor where(const Tensor &cond, const Tensor &x, const Tensor &y);

  // === Cumulative ops ===

  Tensor cumOp(const Tensor &a, OperatorEnum op, std::optional<int> dim = {});

  // === Statistical ops ===

  Tensor variance(const Tensor &a, int correction, std::optional<int> dim = {});

  // === Softmax ===

  Tensor softmax(const Tensor &a, int dim);
  Tensor logSoftmax(const Tensor &a, int dim);

  // === Shape ops ===

  Tensor reshape(const Tensor &a, const std::vector<int32_t> &newShape);
  Tensor squeeze(const Tensor &a, std::optional<int> dim = {});
  Tensor unsqueeze(const Tensor &a, int dim);
  Tensor
  unflatten(const Tensor &a, int dim, const std::vector<uint32_t> &sizes);
  Tensor flatten(const Tensor &a, int startDim, int endDim);

  // === Norm ===

  Tensor norm(const Tensor &a, std::optional<int> dim = {});

  // === Prefix scan ===

  Tensor prefixScan(const Tensor &a, OperatorEnum op);

  // === Convolution ===

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

  // === Pooling ===

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

  // === Normalization ===

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

  // === Embedding ===

  Tensor embedding(const Tensor &indices, const Tensor &weight);

  // === Padding ===

  Tensor pad(const Tensor &input,
             const std::vector<uint32_t> &padWidths,
             float value = 0.0f);

  // === Output marking ===

  void markOutput(const Tensor &t);

private:
  Operations *ops_;
  Graph graph_;
};

} // namespace graph
} // namespace cut
