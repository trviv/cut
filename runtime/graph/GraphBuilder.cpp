#include "GraphBuilder.h"
#include "Runtime.h"

#include <stdexcept>

namespace cut {
namespace graph {

GraphBuilder::GraphBuilder(Runtime &runtime) : runtime_(&runtime) {}

Graph GraphBuilder::build() {
  return std::move(graph_);
}

// ============================================================================
// Helpers
// ============================================================================

const std::vector<uint32_t> &GraphBuilder::getShape(VirtualTensor vt) const {
  return graph_.node(vt).outputShape;
}

DataType GraphBuilder::getDtype(VirtualTensor vt) const {
  return graph_.node(vt).outputDtype;
}

size_t GraphBuilder::shapeProduct(const std::vector<uint32_t> &shape) {
  size_t prod = 1;
  for (uint32_t d : shape)
    prod *= d;
  return prod;
}

GraphBuilder::DimParams
GraphBuilder::computeDimParams(const std::vector<uint32_t> &shape, int dim) {
  int ndim = static_cast<int>(shape.size());
  if (dim < 0)
    dim = ndim + dim;
  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("dim out of range");
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

// ============================================================================
// Graph inputs
// ============================================================================

VirtualTensor GraphBuilder::input(const Tensor &gpuHandle, bool isConstant) {
  const auto &buf = runtime_->getTensor(gpuHandle);
  GraphNode node;
  node.type = GraphNodeType::Input;
  node.outputShape = buf.getShape();
  node.outputDtype = buf.getDtype();
  node.data = InputData{gpuHandle, isConstant};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Element-wise ops
// ============================================================================

VirtualTensor
GraphBuilder::binaryOp(OperatorEnum op, VirtualTensor a, VirtualTensor b) {
  GraphNode node;
  node.type = GraphNodeType::BinaryOp;
  node.inputs = {a, b};
  node.outputShape = getShape(a);
  node.outputDtype = getDtype(a);
  node.data = BinaryOpData{op};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::unaryOp(OperatorEnum op, VirtualTensor a) {
  GraphNode node;
  node.type = GraphNodeType::UnaryOp;
  node.inputs = {a};
  node.outputShape = getShape(a);
  node.outputDtype = getDtype(a);
  node.data = UnaryOpData{op};
  return graph_.addNode(std::move(node));
}

VirtualTensor
GraphBuilder::vecScalarOp(OperatorEnum op, VirtualTensor a, float scalar) {
  GraphNode node;
  node.type = GraphNodeType::VecScalarOp;
  node.inputs = {a};
  node.outputShape = getShape(a);
  node.outputDtype = getDtype(a);
  node.data = VecScalarOpData{op, scalar};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Reduction
// ============================================================================

VirtualTensor
GraphBuilder::reduce(OperatorEnum op, VirtualTensor a, std::optional<int> dim) {
  GraphNode node;
  node.type = GraphNodeType::Reduce;
  node.inputs = {a};
  node.outputDtype = getDtype(a);
  node.data = ReduceData{op, dim};

  if (!dim.has_value()) {
    // Global reduce → shape {1}
    node.outputShape = {1};
  } else {
    auto params = computeDimParams(getShape(a), dim.value());
    node.outputShape = params.outShape;
  }

  return graph_.addNode(std::move(node));
}

// ============================================================================
// Matrix ops
// ============================================================================

VirtualTensor GraphBuilder::matmul(VirtualTensor a, VirtualTensor b) {
  const auto &shapeA = getShape(a);
  const auto &shapeB = getShape(b);
  if (shapeA.size() != 2 || shapeB.size() != 2) {
    throw std::runtime_error("matmul requires 2D matrices");
  }
  if (shapeA[1] != shapeB[0]) {
    throw std::runtime_error("Matrix dimension mismatch");
  }

  GraphNode node;
  node.type = GraphNodeType::MatMul;
  node.inputs = {a, b};
  node.outputShape = {shapeA[0], shapeB[1]};
  // MatMul always outputs Float32 (matching MatMulOpNode::outputDtype)
  node.outputDtype = DataType::Float32;
  node.data = MatMulData{};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::transpose(VirtualTensor a) {
  const auto &shape = getShape(a);
  if (shape.size() != 2) {
    throw std::runtime_error("Transpose requires a 2D tensor");
  }

  GraphNode node;
  node.type = GraphNodeType::Transpose;
  node.inputs = {a};
  node.outputShape = {shape[1], shape[0]};
  node.outputDtype = getDtype(a);
  node.data = TransposeData{};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::dot(VirtualTensor a, VirtualTensor b) {
  GraphNode node;
  node.type = GraphNodeType::Dot;
  node.inputs = {a, b};
  node.outputShape = {1};
  node.outputDtype = DataType::Float32;
  node.data = DotData{};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Special ops
// ============================================================================

VirtualTensor GraphBuilder::clamp(VirtualTensor a, float minVal, float maxVal) {
  GraphNode node;
  node.type = GraphNodeType::Clamp;
  node.inputs = {a};
  node.outputShape = getShape(a);
  node.outputDtype = getDtype(a);
  node.data = ClampData{minVal, maxVal};
  return graph_.addNode(std::move(node));
}

VirtualTensor
GraphBuilder::where(VirtualTensor cond, VirtualTensor x, VirtualTensor y) {
  GraphNode node;
  node.type = GraphNodeType::Where;
  node.inputs = {cond, x, y};
  node.outputShape = getShape(x);
  node.outputDtype = getDtype(x);
  node.data = WhereData{};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Cumulative ops
// ============================================================================

VirtualTensor
GraphBuilder::cumOp(VirtualTensor a, OperatorEnum op, std::optional<int> dim) {
  GraphNode node;
  node.type = GraphNodeType::CumOp;
  node.inputs = {a};
  node.outputShape = getShape(a);
  node.outputDtype = getDtype(a);
  node.data = CumOpData{op, dim};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Statistical ops
// ============================================================================

VirtualTensor GraphBuilder::variance(VirtualTensor a,
                                     int correction,
                                     std::optional<int> dim) {
  GraphNode node;
  node.type = GraphNodeType::Variance;
  node.inputs = {a};
  node.outputDtype = getDtype(a);
  node.data = VarianceData{correction, dim};

  if (!dim.has_value()) {
    node.outputShape = {1};
  } else {
    auto params = computeDimParams(getShape(a), dim.value());
    node.outputShape = params.outShape;
  }

  return graph_.addNode(std::move(node));
}

// ============================================================================
// Softmax
// ============================================================================

VirtualTensor GraphBuilder::softmax(VirtualTensor a, int dim) {
  GraphNode node;
  node.type = GraphNodeType::Softmax;
  node.inputs = {a};
  node.outputShape = getShape(a);
  node.outputDtype = getDtype(a);
  node.data = SoftmaxData{dim};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::logSoftmax(VirtualTensor a, int dim) {
  GraphNode node;
  node.type = GraphNodeType::LogSoftmax;
  node.inputs = {a};
  node.outputShape = getShape(a);
  node.outputDtype = getDtype(a);
  node.data = SoftmaxData{dim};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Shape ops
// ============================================================================

VirtualTensor GraphBuilder::reshape(VirtualTensor a,
                                    const std::vector<int32_t> &newShape) {
  const auto &oldShape = getShape(a);
  size_t oldTotal = shapeProduct(oldShape);

  // Resolve shape (same logic as Operations::reshape)
  std::vector<uint32_t> resolved;
  int negIdx = -1;
  size_t knownTotal = 1;

  for (int i = 0; i < static_cast<int>(newShape.size()); ++i) {
    if (newShape[i] == -1) {
      if (negIdx != -1)
        throw std::invalid_argument("Only one dimension can be -1");
      negIdx = i;
      resolved.push_back(0);
    } else if (newShape[i] < 0) {
      throw std::invalid_argument("Invalid shape dimension");
    } else {
      resolved.push_back(static_cast<uint32_t>(newShape[i]));
      knownTotal *= newShape[i];
    }
  }

  if (negIdx != -1) {
    if (knownTotal == 0)
      throw std::invalid_argument("Cannot infer dimension with zero dims");
    size_t inferred = oldTotal / knownTotal;
    resolved[negIdx] = static_cast<uint32_t>(inferred);
  }

  GraphNode node;
  node.type = GraphNodeType::Reshape;
  node.inputs = {a};
  node.outputShape = resolved;
  node.outputDtype = getDtype(a);
  node.data = ReshapeData{newShape};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::squeeze(VirtualTensor a, std::optional<int> dim) {
  const auto &oldShape = getShape(a);
  int ndim = static_cast<int>(oldShape.size());
  std::vector<uint32_t> newShape;

  if (!dim.has_value()) {
    for (uint32_t s : oldShape) {
      if (s != 1)
        newShape.push_back(s);
    }
  } else {
    int d = dim.value();
    if (d < 0)
      d = ndim + d;
    for (int i = 0; i < ndim; ++i) {
      if (i == d && oldShape[i] == 1)
        continue;
      newShape.push_back(oldShape[i]);
    }
  }
  if (newShape.empty())
    newShape.push_back(1);

  GraphNode node;
  node.type = GraphNodeType::Squeeze;
  node.inputs = {a};
  node.outputShape = newShape;
  node.outputDtype = getDtype(a);
  node.data = SqueezeData{dim};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::unsqueeze(VirtualTensor a, int dim) {
  const auto &oldShape = getShape(a);
  int ndim = static_cast<int>(oldShape.size());

  if (dim < 0)
    dim = ndim + 1 + dim;

  std::vector<uint32_t> newShape(oldShape.begin(), oldShape.begin() + dim);
  newShape.push_back(1);
  newShape.insert(newShape.end(), oldShape.begin() + dim, oldShape.end());

  GraphNode node;
  node.type = GraphNodeType::Unsqueeze;
  node.inputs = {a};
  node.outputShape = newShape;
  node.outputDtype = getDtype(a);
  node.data = UnsqueezeData{dim};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::unflatten(VirtualTensor a,
                                      int dim,
                                      const std::vector<uint32_t> &sizes) {
  const auto &oldShape = getShape(a);
  int ndim = static_cast<int>(oldShape.size());
  if (dim < 0)
    dim = ndim + dim;

  std::vector<uint32_t> newShape(oldShape.begin(), oldShape.begin() + dim);
  newShape.insert(newShape.end(), sizes.begin(), sizes.end());
  newShape.insert(newShape.end(), oldShape.begin() + dim + 1, oldShape.end());

  GraphNode node;
  node.type = GraphNodeType::Unflatten;
  node.inputs = {a};
  node.outputShape = newShape;
  node.outputDtype = getDtype(a);
  node.data = UnflattenData{dim, sizes};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::flatten(VirtualTensor a, int startDim, int endDim) {
  const auto &oldShape = getShape(a);
  int ndim = static_cast<int>(oldShape.size());

  if (endDim < 0)
    endDim = ndim + endDim;
  if (startDim < 0)
    startDim = ndim + startDim;

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

  GraphNode node;
  node.type = GraphNodeType::Flatten;
  node.inputs = {a};
  node.outputShape = newShape;
  node.outputDtype = getDtype(a);
  node.data = FlattenData{startDim, endDim};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Norm
// ============================================================================

VirtualTensor GraphBuilder::norm(VirtualTensor a, std::optional<int> dim) {
  GraphNode node;
  node.type = GraphNodeType::Norm;
  node.inputs = {a};
  node.outputDtype = getDtype(a);
  node.data = NormData{dim};

  if (!dim.has_value()) {
    node.outputShape = {1};
  } else {
    auto params = computeDimParams(getShape(a), dim.value());
    node.outputShape = params.outShape;
  }

  return graph_.addNode(std::move(node));
}

// ============================================================================
// Prefix scan
// ============================================================================

VirtualTensor GraphBuilder::prefixScan(VirtualTensor a, OperatorEnum op) {
  GraphNode node;
  node.type = GraphNodeType::PrefixScan;
  node.inputs = {a};
  node.outputShape = getShape(a);
  node.outputDtype = getDtype(a);
  node.data = PrefixScanData{op};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Convolution
// ============================================================================

VirtualTensor GraphBuilder::conv1d(VirtualTensor input,
                                   VirtualTensor weight,
                                   uint32_t stride,
                                   uint32_t padding) {
  const auto &inShape = getShape(input);
  const auto &wShape = getShape(weight);
  // Input: [channels, length], Weight: [out_channels, in_channels, kernel]
  uint32_t outChannels = wShape[0];
  uint32_t inLen = inShape.back();
  uint32_t kernel = wShape.back();
  uint32_t outLen = (inLen + 2 * padding - kernel) / stride + 1;

  GraphNode node;
  node.type = GraphNodeType::Conv1d;
  node.inputs = {input, weight};
  node.outputShape = {outChannels, outLen};
  node.outputDtype = getDtype(input);
  node.data = Conv1dData{stride, padding};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::conv2d(VirtualTensor input,
                                   VirtualTensor weight,
                                   uint32_t strideH,
                                   uint32_t strideW,
                                   uint32_t padH,
                                   uint32_t padW) {
  const auto &inShape = getShape(input);
  const auto &wShape = getShape(weight);
  // Input: [N, C, H, W], Weight: [outC, inC, kH, kW]
  uint32_t N = inShape[0];
  uint32_t outC = wShape[0];
  uint32_t H = inShape[2], W = inShape[3];
  uint32_t kH = wShape[2], kW = wShape[3];
  uint32_t outH = (H + 2 * padH - kH) / strideH + 1;
  uint32_t outW = (W + 2 * padW - kW) / strideW + 1;

  GraphNode node;
  node.type = GraphNodeType::Conv2d;
  node.inputs = {input, weight};
  node.outputShape = {N, outC, outH, outW};
  node.outputDtype = getDtype(input);
  node.data = Conv2dData{strideH, strideW, padH, padW};
  return graph_.addNode(std::move(node));
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
  const auto &inShape = getShape(input);
  uint32_t N = inShape[0], C = inShape[1];
  uint32_t H = inShape[2], W = inShape[3];
  uint32_t outH = (H + 2 * padH - kernelH) / strideH + 1;
  uint32_t outW = (W + 2 * padW - kernelW) / strideW + 1;

  GraphNode node;
  node.type = GraphNodeType::MaxPool2d;
  node.inputs = {input};
  node.outputShape = {N, C, outH, outW};
  node.outputDtype = getDtype(input);
  node.data = Pool2dData{kernelH, kernelW, strideH, strideW, padH, padW};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::avgPool2d(VirtualTensor input,
                                      uint32_t kernelH,
                                      uint32_t kernelW,
                                      uint32_t strideH,
                                      uint32_t strideW,
                                      uint32_t padH,
                                      uint32_t padW) {
  const auto &inShape = getShape(input);
  uint32_t N = inShape[0], C = inShape[1];
  uint32_t H = inShape[2], W = inShape[3];
  uint32_t outH = (H + 2 * padH - kernelH) / strideH + 1;
  uint32_t outW = (W + 2 * padW - kernelW) / strideW + 1;

  GraphNode node;
  node.type = GraphNodeType::AvgPool2d;
  node.inputs = {input};
  node.outputShape = {N, C, outH, outW};
  node.outputDtype = getDtype(input);
  node.data = Pool2dData{kernelH, kernelW, strideH, strideW, padH, padW};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::adaptiveAvgPool2d(VirtualTensor input,
                                              uint32_t outH,
                                              uint32_t outW) {
  const auto &inShape = getShape(input);
  uint32_t N = inShape[0], C = inShape[1];

  GraphNode node;
  node.type = GraphNodeType::AdaptiveAvgPool2d;
  node.inputs = {input};
  node.outputShape = {N, C, outH, outW};
  node.outputDtype = getDtype(input);
  node.data = AdaptivePool2dData{outH, outW};
  return graph_.addNode(std::move(node));
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
  GraphNode node;
  node.type = GraphNodeType::LayerNorm;
  node.inputs = {input};
  if (weight)
    node.inputs.push_back(*weight);
  if (bias)
    node.inputs.push_back(*bias);
  node.outputShape = getShape(input);
  node.outputDtype = getDtype(input);
  node.data =
      LayerNormData{normalizedShape, eps, weight != nullptr, bias != nullptr};
  return graph_.addNode(std::move(node));
}

VirtualTensor GraphBuilder::batchNorm(VirtualTensor input,
                                      VirtualTensor runningMean,
                                      VirtualTensor runningVar,
                                      VirtualTensor *weight,
                                      VirtualTensor *bias,
                                      float eps) {
  GraphNode node;
  node.type = GraphNodeType::BatchNorm;
  node.inputs = {input, runningMean, runningVar};
  if (weight)
    node.inputs.push_back(*weight);
  if (bias)
    node.inputs.push_back(*bias);
  node.outputShape = getShape(input);
  node.outputDtype = getDtype(input);
  node.data = BatchNormData{eps, weight != nullptr, bias != nullptr};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Embedding
// ============================================================================

VirtualTensor GraphBuilder::embedding(VirtualTensor indices,
                                      VirtualTensor weight) {
  const auto &idxShape = getShape(indices);
  const auto &wShape = getShape(weight);
  // indices: [numIndices], weight: [vocabSize, embDim]
  // output: [numIndices, embDim]
  uint32_t numIndices = 1;
  for (uint32_t d : idxShape)
    numIndices *= d;
  uint32_t embDim = wShape.back();

  GraphNode node;
  node.type = GraphNodeType::Embedding;
  node.inputs = {indices, weight};
  node.outputShape = {numIndices, embDim};
  node.outputDtype = getDtype(weight);
  node.data = EmbeddingData{};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Padding
// ============================================================================

VirtualTensor GraphBuilder::pad(VirtualTensor input,
                                const std::vector<uint32_t> &padWidths,
                                float value) {
  const auto &inShape = getShape(input);
  uint32_t ndim = static_cast<uint32_t>(inShape.size());
  std::vector<uint32_t> outShape(ndim);
  for (uint32_t i = 0; i < ndim; ++i) {
    uint32_t before = (i < padWidths.size() / 2) ? padWidths[2 * i] : 0;
    uint32_t after = (i < padWidths.size() / 2) ? padWidths[2 * i + 1] : 0;
    outShape[i] = inShape[i] + before + after;
  }

  GraphNode node;
  node.type = GraphNodeType::Pad;
  node.inputs = {input};
  node.outputShape = outShape;
  node.outputDtype = getDtype(input);
  node.data = PadData{padWidths, value};
  return graph_.addNode(std::move(node));
}

// ============================================================================
// Output marking
// ============================================================================

void GraphBuilder::markOutput(VirtualTensor vt) {
  graph_.markOutput(vt);
}

} // namespace graph
} // namespace cut
