#include "GraphBuilder.h"
#include "OpNode.h"
#include "Operations.h"
#include "Runtime.h"

#include <cstring>
#include <stdexcept>

namespace cut {
namespace graph {

GraphBuilder::GraphBuilder(Runtime &runtime) : ops_(&runtime.ops()) {
  ops_->setGraph(&graph_);
}

GraphBuilder::~GraphBuilder() {
  // Ensure we exit graph mode if build() was never called
  if (ops_->isGraphMode()) {
    ops_->clearGraph();
  }
}

Graph GraphBuilder::build() {
  ops_->clearGraph();
  return std::move(graph_);
}

// ============================================================================
// Helpers
// ============================================================================

const Tensor &GraphBuilder::resolve(VirtualTensor vt) const {
  return vtToTensor_[vt.id];
}

VirtualTensor GraphBuilder::mapResult(const Tensor &t) {
  // The Operations layer creates a mapping from Tensor -> VirtualTensor
  // when in graph mode. Find the VirtualTensor for this result.
  const auto &mappings = ops_->graphMappings();
  for (auto it = mappings.rbegin(); it != mappings.rend(); ++it) {
    if (it->first == t) {
      // Ensure vtToTensor_ is large enough
      if (it->second.id >= vtToTensor_.size()) {
        vtToTensor_.resize(it->second.id + 1);
      }
      vtToTensor_[it->second.id] = t;
      return it->second;
    }
  }
  throw std::runtime_error("GraphBuilder: failed to map result tensor");
}

// ============================================================================
// Graph inputs
// ============================================================================

VirtualTensor GraphBuilder::input(const Tensor &gpuHandle, bool isConstant) {
  VirtualTensor vt = ops_->registerInput(gpuHandle, isConstant);
  if (vt.id >= vtToTensor_.size()) {
    vtToTensor_.resize(vt.id + 1);
  }
  vtToTensor_[vt.id] = gpuHandle;
  return vt;
}

// ============================================================================
// Element-wise ops
// ============================================================================

VirtualTensor
GraphBuilder::binaryOp(OperatorEnum op, VirtualTensor a, VirtualTensor b) {
  Tensor result = ops_->binaryOp(op, resolve(a), resolve(b));
  return mapResult(result);
}

VirtualTensor GraphBuilder::unaryOp(OperatorEnum op, VirtualTensor a) {
  Tensor result = ops_->unaryOp(op, resolve(a));
  return mapResult(result);
}

VirtualTensor
GraphBuilder::vecScalarOp(OperatorEnum op, VirtualTensor a, float scalar) {
  Tensor result =
      ops_->vecScalarOp(op, resolve(a), DataReference(&scalar, sizeof(float)));
  return mapResult(result);
}

// ============================================================================
// Reduction
// ============================================================================

VirtualTensor
GraphBuilder::reduce(OperatorEnum op, VirtualTensor a, std::optional<int> dim) {
  Tensor result = ops_->reduce(op, resolve(a), dim);
  return mapResult(result);
}

// ============================================================================
// Matrix ops
// ============================================================================

VirtualTensor GraphBuilder::matmul(VirtualTensor a, VirtualTensor b) {
  Tensor result = ops_->matmul(resolve(a), resolve(b));
  return mapResult(result);
}

VirtualTensor GraphBuilder::transpose(VirtualTensor a) {
  Tensor result = ops_->transpose(resolve(a));
  return mapResult(result);
}

VirtualTensor GraphBuilder::dot(VirtualTensor a, VirtualTensor b) {
  Tensor result = ops_->dot(resolve(a), resolve(b));
  return mapResult(result);
}

// ============================================================================
// Special ops
// ============================================================================

VirtualTensor GraphBuilder::clamp(VirtualTensor a, float minVal, float maxVal) {
  float vals[2] = {minVal, maxVal};
  Tensor result = ops_->clamp(resolve(a), DataReference(vals, 8));
  return mapResult(result);
}

VirtualTensor
GraphBuilder::where(VirtualTensor cond, VirtualTensor x, VirtualTensor y) {
  Tensor result = ops_->where(resolve(cond), resolve(x), resolve(y));
  return mapResult(result);
}

// ============================================================================
// Cumulative ops
// ============================================================================

VirtualTensor
GraphBuilder::cumOp(VirtualTensor a, OperatorEnum op, std::optional<int> dim) {
  Tensor result = ops_->cumOp(resolve(a), op, dim);
  return mapResult(result);
}

// ============================================================================
// Statistical ops
// ============================================================================

VirtualTensor GraphBuilder::variance(VirtualTensor a,
                                     int correction,
                                     std::optional<int> dim) {
  Tensor result = ops_->variance(resolve(a), correction, dim);
  return mapResult(result);
}

// ============================================================================
// Softmax
// ============================================================================

VirtualTensor GraphBuilder::softmax(VirtualTensor a, int dim) {
  Tensor result = ops_->softmax(resolve(a), dim);
  return mapResult(result);
}

VirtualTensor GraphBuilder::logSoftmax(VirtualTensor a, int dim) {
  Tensor result = ops_->logSoftmax(resolve(a), dim);
  return mapResult(result);
}

// ============================================================================
// Shape ops
// ============================================================================

VirtualTensor GraphBuilder::reshape(VirtualTensor a,
                                    const std::vector<int32_t> &newShape) {
  Tensor result = ops_->reshape(resolve(a), newShape);
  return mapResult(result);
}

VirtualTensor GraphBuilder::squeeze(VirtualTensor a, std::optional<int> dim) {
  Tensor result = ops_->squeeze(resolve(a), dim);
  return mapResult(result);
}

VirtualTensor GraphBuilder::unsqueeze(VirtualTensor a, int dim) {
  Tensor result = ops_->unsqueeze(resolve(a), dim);
  return mapResult(result);
}

VirtualTensor GraphBuilder::unflatten(VirtualTensor a,
                                      int dim,
                                      const std::vector<uint32_t> &sizes) {
  Tensor result = ops_->unflatten(resolve(a), dim, sizes);
  return mapResult(result);
}

VirtualTensor GraphBuilder::flatten(VirtualTensor a, int startDim, int endDim) {
  Tensor result = ops_->flatten(resolve(a), startDim, endDim);
  return mapResult(result);
}

// ============================================================================
// Norm
// ============================================================================

VirtualTensor GraphBuilder::norm(VirtualTensor a, std::optional<int> dim) {
  Tensor result = ops_->norm(resolve(a), dim);
  return mapResult(result);
}

// ============================================================================
// Prefix scan
// ============================================================================

VirtualTensor GraphBuilder::prefixScan(VirtualTensor a, OperatorEnum op) {
  Tensor result = ops_->prefixScan(resolve(a), op);
  return mapResult(result);
}

// ============================================================================
// Convolution
// ============================================================================

VirtualTensor GraphBuilder::conv1d(VirtualTensor input,
                                   VirtualTensor weight,
                                   uint32_t stride,
                                   uint32_t padding) {
  Tensor result =
      ops_->conv1d(resolve(input), resolve(weight), stride, padding);
  return mapResult(result);
}

VirtualTensor GraphBuilder::conv2d(VirtualTensor input,
                                   VirtualTensor weight,
                                   uint32_t strideH,
                                   uint32_t strideW,
                                   uint32_t padH,
                                   uint32_t padW) {
  Tensor result = ops_->conv2d(resolve(input), resolve(weight), strideH,
                               strideW, padH, padW);
  return mapResult(result);
}

// ============================================================================
// Pooling
// ============================================================================

VirtualTensor GraphBuilder::maxPool2d(VirtualTensor input,
                                      uint32_t kernelH,
                                      uint32_t kernelW,
                                      uint32_t strideH,
                                      uint32_t strideW,
                                      uint32_t padH,
                                      uint32_t padW) {
  Tensor result = ops_->maxPool2d(resolve(input), kernelH, kernelW, strideH,
                                  strideW, padH, padW);
  return mapResult(result);
}

VirtualTensor GraphBuilder::avgPool2d(VirtualTensor input,
                                      uint32_t kernelH,
                                      uint32_t kernelW,
                                      uint32_t strideH,
                                      uint32_t strideW,
                                      uint32_t padH,
                                      uint32_t padW) {
  Tensor result = ops_->avgPool2d(resolve(input), kernelH, kernelW, strideH,
                                  strideW, padH, padW);
  return mapResult(result);
}

VirtualTensor GraphBuilder::adaptiveAvgPool2d(VirtualTensor input,
                                              uint32_t outH,
                                              uint32_t outW) {
  Tensor result = ops_->adaptiveAvgPool2d(resolve(input), outH, outW);
  return mapResult(result);
}

// ============================================================================
// Normalization
// ============================================================================

VirtualTensor
GraphBuilder::layerNorm(VirtualTensor input,
                        const std::vector<uint32_t> &normalizedShape,
                        VirtualTensor *weight,
                        VirtualTensor *bias,
                        float eps) {
  const Tensor *wPtr = nullptr;
  const Tensor *bPtr = nullptr;
  Tensor wTensor, bTensor;
  if (weight) {
    wTensor = resolve(*weight);
    wPtr = &wTensor;
  }
  if (bias) {
    bTensor = resolve(*bias);
    bPtr = &bTensor;
  }
  Tensor result =
      ops_->layerNorm(resolve(input), normalizedShape, wPtr, bPtr, eps);
  return mapResult(result);
}

VirtualTensor GraphBuilder::batchNorm(VirtualTensor input,
                                      VirtualTensor runningMean,
                                      VirtualTensor runningVar,
                                      VirtualTensor *weight,
                                      VirtualTensor *bias,
                                      float eps) {
  const Tensor *wPtr = nullptr;
  const Tensor *bPtr = nullptr;
  Tensor wTensor, bTensor;
  if (weight) {
    wTensor = resolve(*weight);
    wPtr = &wTensor;
  }
  if (bias) {
    bTensor = resolve(*bias);
    bPtr = &bTensor;
  }
  Tensor result = ops_->batchNorm(resolve(input), resolve(runningMean),
                                  resolve(runningVar), wPtr, bPtr, eps);
  return mapResult(result);
}

// ============================================================================
// Embedding
// ============================================================================

VirtualTensor GraphBuilder::embedding(VirtualTensor indices,
                                      VirtualTensor weight) {
  Tensor result = ops_->embedding(resolve(indices), resolve(weight));
  return mapResult(result);
}

// ============================================================================
// Padding
// ============================================================================

VirtualTensor GraphBuilder::pad(VirtualTensor input,
                                const std::vector<uint32_t> &padWidths,
                                float value) {
  Tensor result = ops_->pad(resolve(input), padWidths, value);
  return mapResult(result);
}

// ============================================================================
// Output marking
// ============================================================================

void GraphBuilder::markOutput(VirtualTensor vt) {
  graph_.markOutput(vt);
}

} // namespace graph
} // namespace cut
