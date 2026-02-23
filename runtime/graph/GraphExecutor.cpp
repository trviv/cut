#include "GraphExecutor.h"
#include "Operations.h"
#include "Runtime.h"

#include <cstring>
#include <stdexcept>

namespace cut {
namespace graph {

GraphExecutor::GraphExecutor(Operations &ops, Runtime &runtime)
    : ops_(&ops), runtime_(&runtime) {}

std::vector<Tensor> GraphExecutor::execute(const Graph &graph) {
  tensorMap_.clear();
  tensorMap_.resize(graph.size());

  // Execute in topological order
  auto order = graph.topologicalOrder();
  for (uint32_t idx : order) {
    executeNode(graph, idx);
  }

  // Collect outputs
  std::vector<Tensor> results;
  for (const auto &out : graph.outputs()) {
    if (out.isValid()) {
      results.push_back(tensorMap_[out.id]);
    }
  }
  return results;
}

void GraphExecutor::executeNode(const Graph &graph, uint32_t nodeIndex) {
  const auto &n = graph.nodes()[nodeIndex];
  if (n.isRemoved)
    return;

  // Helper to get real tensor for a VirtualTensor input
  auto real = [&](size_t inputIdx) -> Tensor {
    return tensorMap_[n.inputs[inputIdx].id];
  };

  switch (n.type) {
  case GraphNodeType::Input: {
    const auto &d = std::get<InputData>(n.data);
    tensorMap_[nodeIndex] = d.gpuHandle;
    break;
  }

  case GraphNodeType::BinaryOp: {
    const auto &d = std::get<BinaryOpData>(n.data);
    tensorMap_[nodeIndex] = ops_->binaryOp(d.op, real(0), real(1));
    break;
  }

  case GraphNodeType::UnaryOp: {
    const auto &d = std::get<UnaryOpData>(n.data);
    tensorMap_[nodeIndex] = ops_->unaryOp(d.op, real(0));
    break;
  }

  case GraphNodeType::VecScalarOp: {
    const auto &d = std::get<VecScalarOpData>(n.data);
    tensorMap_[nodeIndex] = ops_->vecScalarOp(d.op, real(0), d.scalar);
    break;
  }

  case GraphNodeType::Reduce: {
    const auto &d = std::get<ReduceData>(n.data);
    tensorMap_[nodeIndex] = ops_->reduce(d.op, real(0), d.dim);
    break;
  }

  case GraphNodeType::MatMul: {
    tensorMap_[nodeIndex] = ops_->matmul(real(0), real(1));
    break;
  }

  case GraphNodeType::Transpose: {
    tensorMap_[nodeIndex] = ops_->transpose(real(0));
    break;
  }

  case GraphNodeType::Dot: {
    tensorMap_[nodeIndex] = ops_->dot(real(0), real(1));
    break;
  }

  case GraphNodeType::Clamp: {
    const auto &d = std::get<ClampData>(n.data);
    float vals[2] = {d.minVal, d.maxVal};
    tensorMap_[nodeIndex] = ops_->clamp(real(0), DataReference(vals, 8));
    break;
  }

  case GraphNodeType::Where: {
    tensorMap_[nodeIndex] = ops_->where(real(0), real(1), real(2));
    break;
  }

  case GraphNodeType::CumOp: {
    const auto &d = std::get<CumOpData>(n.data);
    tensorMap_[nodeIndex] = ops_->cumOp(real(0), d.op, d.dim);
    break;
  }

  case GraphNodeType::Variance: {
    const auto &d = std::get<VarianceData>(n.data);
    tensorMap_[nodeIndex] = ops_->variance(real(0), d.correction, d.dim);
    break;
  }

  case GraphNodeType::Softmax: {
    const auto &d = std::get<SoftmaxData>(n.data);
    tensorMap_[nodeIndex] = ops_->softmax(real(0), d.dim);
    break;
  }

  case GraphNodeType::LogSoftmax: {
    const auto &d = std::get<SoftmaxData>(n.data);
    tensorMap_[nodeIndex] = ops_->logSoftmax(real(0), d.dim);
    break;
  }

  case GraphNodeType::Reshape: {
    const auto &d = std::get<ReshapeData>(n.data);
    tensorMap_[nodeIndex] = ops_->reshape(real(0), d.newShape);
    break;
  }

  case GraphNodeType::Squeeze: {
    const auto &d = std::get<SqueezeData>(n.data);
    tensorMap_[nodeIndex] = ops_->squeeze(real(0), d.dim);
    break;
  }

  case GraphNodeType::Unsqueeze: {
    const auto &d = std::get<UnsqueezeData>(n.data);
    tensorMap_[nodeIndex] = ops_->unsqueeze(real(0), d.dim);
    break;
  }

  case GraphNodeType::Unflatten: {
    const auto &d = std::get<UnflattenData>(n.data);
    tensorMap_[nodeIndex] = ops_->unflatten(real(0), d.dim, d.sizes);
    break;
  }

  case GraphNodeType::Flatten: {
    const auto &d = std::get<FlattenData>(n.data);
    tensorMap_[nodeIndex] = ops_->flatten(real(0), d.startDim, d.endDim);
    break;
  }

  case GraphNodeType::Norm: {
    const auto &d = std::get<NormData>(n.data);
    tensorMap_[nodeIndex] = ops_->norm(real(0), d.dim);
    break;
  }

  case GraphNodeType::PrefixScan: {
    const auto &d = std::get<PrefixScanData>(n.data);
    tensorMap_[nodeIndex] = ops_->prefixScan(real(0), d.op);
    break;
  }

  case GraphNodeType::Conv1d: {
    const auto &d = std::get<Conv1dData>(n.data);
    tensorMap_[nodeIndex] = ops_->conv1d(real(0), real(1), d.stride, d.padding);
    break;
  }

  case GraphNodeType::Conv2d: {
    const auto &d = std::get<Conv2dData>(n.data);
    tensorMap_[nodeIndex] =
        ops_->conv2d(real(0), real(1), d.strideH, d.strideW, d.padH, d.padW);
    break;
  }

  case GraphNodeType::MaxPool2d: {
    const auto &d = std::get<Pool2dData>(n.data);
    tensorMap_[nodeIndex] = ops_->maxPool2d(
        real(0), d.kernelH, d.kernelW, d.strideH, d.strideW, d.padH, d.padW);
    break;
  }

  case GraphNodeType::AvgPool2d: {
    const auto &d = std::get<Pool2dData>(n.data);
    tensorMap_[nodeIndex] = ops_->avgPool2d(
        real(0), d.kernelH, d.kernelW, d.strideH, d.strideW, d.padH, d.padW);
    break;
  }

  case GraphNodeType::AdaptiveAvgPool2d: {
    const auto &d = std::get<AdaptivePool2dData>(n.data);
    tensorMap_[nodeIndex] = ops_->adaptiveAvgPool2d(real(0), d.outH, d.outW);
    break;
  }

  case GraphNodeType::LayerNorm: {
    const auto &d = std::get<LayerNormData>(n.data);
    // inputs: [input, weight?, bias?]
    Tensor *wPtr = nullptr;
    Tensor *bPtr = nullptr;
    Tensor wTensor, bTensor;
    size_t nextInput = 1;
    if (d.hasWeight) {
      wTensor = real(nextInput++);
      wPtr = &wTensor;
    }
    if (d.hasBias) {
      bTensor = real(nextInput++);
      bPtr = &bTensor;
    }
    tensorMap_[nodeIndex] =
        ops_->layerNorm(real(0), d.normalizedShape, wPtr, bPtr, d.eps);
    break;
  }

  case GraphNodeType::BatchNorm: {
    const auto &d = std::get<BatchNormData>(n.data);
    // inputs: [input, runningMean, runningVar, weight?, bias?]
    Tensor *wPtr = nullptr;
    Tensor *bPtr = nullptr;
    Tensor wTensor, bTensor;
    size_t nextInput = 3;
    if (d.hasWeight) {
      wTensor = real(nextInput++);
      wPtr = &wTensor;
    }
    if (d.hasBias) {
      bTensor = real(nextInput++);
      bPtr = &bTensor;
    }
    tensorMap_[nodeIndex] =
        ops_->batchNorm(real(0), real(1), real(2), wPtr, bPtr, d.eps);
    break;
  }

  case GraphNodeType::Embedding: {
    tensorMap_[nodeIndex] = ops_->embedding(real(0), real(1));
    break;
  }

  case GraphNodeType::Pad: {
    const auto &d = std::get<PadData>(n.data);
    tensorMap_[nodeIndex] = ops_->pad(real(0), d.padWidths, d.value);
    break;
  }
  }
}

} // namespace graph
} // namespace cut
