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
// Graph inputs
// ============================================================================

Tensor GraphBuilder::input(const Tensor &gpuHandle, bool isConstant) {
  return ops_->registerInput(gpuHandle, isConstant);
}

// ============================================================================
// Element-wise ops
// ============================================================================

Tensor
GraphBuilder::binaryOp(OperatorEnum op, const Tensor &a, const Tensor &b) {
  return ops_->binaryOp(op, a, b);
}

Tensor GraphBuilder::unaryOp(OperatorEnum op, const Tensor &a) {
  return ops_->unaryOp(op, a);
}

Tensor
GraphBuilder::vecScalarOp(OperatorEnum op, const Tensor &a, float scalar) {
  return ops_->vecScalarOp(op, a, DataReference(&scalar, sizeof(float)));
}

// ============================================================================
// Reduction
// ============================================================================

Tensor
GraphBuilder::reduce(OperatorEnum op, const Tensor &a, std::optional<int> dim) {
  return ops_->reduce(op, a, dim);
}

// ============================================================================
// Matrix ops
// ============================================================================

Tensor GraphBuilder::matmul(const Tensor &a, const Tensor &b) {
  return ops_->matmul(a, b);
}

Tensor GraphBuilder::transpose(const Tensor &a) {
  return ops_->transpose(a);
}

Tensor GraphBuilder::dot(const Tensor &a, const Tensor &b) {
  return ops_->dot(a, b);
}

// ============================================================================
// Special ops
// ============================================================================

Tensor GraphBuilder::clamp(const Tensor &a, float minVal, float maxVal) {
  float vals[2] = {minVal, maxVal};
  return ops_->clamp(a, DataReference(vals, 8));
}

Tensor
GraphBuilder::where(const Tensor &cond, const Tensor &x, const Tensor &y) {
  return ops_->where(cond, x, y);
}

// ============================================================================
// Cumulative ops
// ============================================================================

Tensor
GraphBuilder::cumOp(const Tensor &a, OperatorEnum op, std::optional<int> dim) {
  return ops_->cumOp(a, op, dim);
}

// ============================================================================
// Statistical ops
// ============================================================================

Tensor GraphBuilder::variance(const Tensor &a,
                              int correction,
                              std::optional<int> dim) {
  return ops_->variance(a, correction, dim);
}

// ============================================================================
// Softmax
// ============================================================================

Tensor GraphBuilder::softmax(const Tensor &a, int dim) {
  return ops_->softmax(a, dim);
}

Tensor GraphBuilder::logSoftmax(const Tensor &a, int dim) {
  return ops_->logSoftmax(a, dim);
}

// ============================================================================
// Shape ops
// ============================================================================

Tensor GraphBuilder::reshape(const Tensor &a,
                             const std::vector<int32_t> &newShape) {
  return ops_->reshape(a, newShape);
}

Tensor GraphBuilder::squeeze(const Tensor &a, std::optional<int> dim) {
  return ops_->squeeze(a, dim);
}

Tensor GraphBuilder::unsqueeze(const Tensor &a, int dim) {
  return ops_->unsqueeze(a, dim);
}

Tensor GraphBuilder::unflatten(const Tensor &a,
                               int dim,
                               const std::vector<uint32_t> &sizes) {
  return ops_->unflatten(a, dim, sizes);
}

Tensor GraphBuilder::flatten(const Tensor &a, int startDim, int endDim) {
  return ops_->flatten(a, startDim, endDim);
}

// ============================================================================
// Norm
// ============================================================================

Tensor GraphBuilder::norm(const Tensor &a, std::optional<int> dim) {
  return ops_->norm(a, dim);
}

// ============================================================================
// Prefix scan
// ============================================================================

Tensor GraphBuilder::prefixScan(const Tensor &a, OperatorEnum op) {
  return ops_->prefixScan(a, op);
}

// ============================================================================
// Convolution
// ============================================================================

Tensor GraphBuilder::conv1d(const Tensor &input,
                            const Tensor &weight,
                            uint32_t stride,
                            uint32_t padding) {
  return ops_->conv1d(input, weight, stride, padding);
}

Tensor GraphBuilder::conv2d(const Tensor &input,
                            const Tensor &weight,
                            uint32_t strideH,
                            uint32_t strideW,
                            uint32_t padH,
                            uint32_t padW) {
  return ops_->conv2d(input, weight, strideH, strideW, padH, padW);
}

// ============================================================================
// Pooling
// ============================================================================

Tensor GraphBuilder::maxPool2d(const Tensor &input,
                               uint32_t kernelH,
                               uint32_t kernelW,
                               uint32_t strideH,
                               uint32_t strideW,
                               uint32_t padH,
                               uint32_t padW) {
  return ops_->maxPool2d(input, kernelH, kernelW, strideH, strideW, padH, padW);
}

Tensor GraphBuilder::avgPool2d(const Tensor &input,
                               uint32_t kernelH,
                               uint32_t kernelW,
                               uint32_t strideH,
                               uint32_t strideW,
                               uint32_t padH,
                               uint32_t padW) {
  return ops_->avgPool2d(input, kernelH, kernelW, strideH, strideW, padH, padW);
}

Tensor GraphBuilder::adaptiveAvgPool2d(const Tensor &input,
                                       uint32_t outH,
                                       uint32_t outW) {
  return ops_->adaptiveAvgPool2d(input, outH, outW);
}

// ============================================================================
// Normalization
// ============================================================================

Tensor GraphBuilder::layerNorm(const Tensor &input,
                               const std::vector<uint32_t> &normalizedShape,
                               const Tensor *weight,
                               const Tensor *bias,
                               float eps) {
  return ops_->layerNorm(input, normalizedShape, weight, bias, eps);
}

Tensor GraphBuilder::batchNorm(const Tensor &input,
                               const Tensor &runningMean,
                               const Tensor &runningVar,
                               const Tensor *weight,
                               const Tensor *bias,
                               float eps) {
  return ops_->batchNorm(input, runningMean, runningVar, weight, bias, eps);
}

// ============================================================================
// Embedding
// ============================================================================

Tensor GraphBuilder::embedding(const Tensor &indices, const Tensor &weight) {
  return ops_->embedding(indices, weight);
}

// ============================================================================
// Padding
// ============================================================================

Tensor GraphBuilder::pad(const Tensor &input,
                         const std::vector<uint32_t> &padWidths,
                         float value) {
  return ops_->pad(input, padWidths, value);
}

// ============================================================================
// Output marking
// ============================================================================

void GraphBuilder::markOutput(const Tensor &t) {
  graph_.markOutput(t);
}

} // namespace graph
} // namespace cut
