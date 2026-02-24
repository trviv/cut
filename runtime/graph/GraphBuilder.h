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
/// Operations and maps the resulting Tensor to a VirtualTensor. After
/// construction, call build() to obtain the Graph for optimization and
/// execution.
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
  VirtualTensor input(const Tensor &gpuHandle, bool isConstant = false);

  // === Element-wise ops ===

  VirtualTensor binaryOp(OperatorEnum op, VirtualTensor a, VirtualTensor b);
  VirtualTensor unaryOp(OperatorEnum op, VirtualTensor a);
  VirtualTensor vecScalarOp(OperatorEnum op, VirtualTensor a, float scalar);

  // === Reduction ===

  VirtualTensor
  reduce(OperatorEnum op, VirtualTensor a, std::optional<int> dim = {});

  // === Matrix ops ===

  VirtualTensor matmul(VirtualTensor a, VirtualTensor b);
  VirtualTensor transpose(VirtualTensor a);
  VirtualTensor dot(VirtualTensor a, VirtualTensor b);

  // === Special ops ===

  VirtualTensor clamp(VirtualTensor a, float minVal, float maxVal);
  VirtualTensor where(VirtualTensor cond, VirtualTensor x, VirtualTensor y);

  // === Cumulative ops ===

  VirtualTensor
  cumOp(VirtualTensor a, OperatorEnum op, std::optional<int> dim = {});

  // === Statistical ops ===

  VirtualTensor
  variance(VirtualTensor a, int correction, std::optional<int> dim = {});

  // === Softmax ===

  VirtualTensor softmax(VirtualTensor a, int dim);
  VirtualTensor logSoftmax(VirtualTensor a, int dim);

  // === Shape ops ===

  VirtualTensor reshape(VirtualTensor a, const std::vector<int32_t> &newShape);
  VirtualTensor squeeze(VirtualTensor a, std::optional<int> dim = {});
  VirtualTensor unsqueeze(VirtualTensor a, int dim);
  VirtualTensor
  unflatten(VirtualTensor a, int dim, const std::vector<uint32_t> &sizes);
  VirtualTensor flatten(VirtualTensor a, int startDim, int endDim);

  // === Norm ===

  VirtualTensor norm(VirtualTensor a, std::optional<int> dim = {});

  // === Prefix scan ===

  VirtualTensor prefixScan(VirtualTensor a, OperatorEnum op);

  // === Convolution ===

  VirtualTensor conv1d(VirtualTensor input,
                       VirtualTensor weight,
                       uint32_t stride = 1,
                       uint32_t padding = 0);
  VirtualTensor conv2d(VirtualTensor input,
                       VirtualTensor weight,
                       uint32_t strideH = 1,
                       uint32_t strideW = 1,
                       uint32_t padH = 0,
                       uint32_t padW = 0);

  // === Pooling ===

  VirtualTensor maxPool2d(VirtualTensor input,
                          uint32_t kernelH,
                          uint32_t kernelW,
                          uint32_t strideH = 1,
                          uint32_t strideW = 1,
                          uint32_t padH = 0,
                          uint32_t padW = 0);
  VirtualTensor avgPool2d(VirtualTensor input,
                          uint32_t kernelH,
                          uint32_t kernelW,
                          uint32_t strideH = 1,
                          uint32_t strideW = 1,
                          uint32_t padH = 0,
                          uint32_t padW = 0);
  VirtualTensor
  adaptiveAvgPool2d(VirtualTensor input, uint32_t outH, uint32_t outW);

  // === Normalization ===

  VirtualTensor layerNorm(VirtualTensor input,
                          const std::vector<uint32_t> &normalizedShape,
                          VirtualTensor *weight = nullptr,
                          VirtualTensor *bias = nullptr,
                          float eps = 1e-5f);
  VirtualTensor batchNorm(VirtualTensor input,
                          VirtualTensor runningMean,
                          VirtualTensor runningVar,
                          VirtualTensor *weight = nullptr,
                          VirtualTensor *bias = nullptr,
                          float eps = 1e-5f);

  // === Embedding ===

  VirtualTensor embedding(VirtualTensor indices, VirtualTensor weight);

  // === Padding ===

  VirtualTensor pad(VirtualTensor input,
                    const std::vector<uint32_t> &padWidths,
                    float value = 0.0f);

  // === Output marking ===

  void markOutput(VirtualTensor vt);

private:
  Operations *ops_;
  Graph graph_;

  /// Maps VirtualTensor id → placeholder Tensor for Operations calls.
  std::vector<Tensor> vtToTensor_;

  /// Resolve a VirtualTensor to its placeholder Tensor.
  const Tensor &resolve(VirtualTensor vt) const;

  /// Map an Operations result Tensor back to a VirtualTensor.
  VirtualTensor mapResult(const Tensor &t);
};

} // namespace graph
} // namespace cut
