#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeOps.h>

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace cut {
namespace graph {

// ============================================================================
// VirtualTensor — lightweight proxy for a tensor that doesn't exist on GPU yet
// ============================================================================

/// Index into Graph::nodes_. Represents the output of one graph node.
struct VirtualTensor {
  uint32_t id = UINT32_MAX;

  bool isValid() const { return id != UINT32_MAX; }
  bool operator==(const VirtualTensor &o) const { return id == o.id; }
  bool operator!=(const VirtualTensor &o) const { return id != o.id; }
};

// ============================================================================
// GraphNodeType — identifies which operation a graph node represents
// ============================================================================

enum class GraphNodeType {
  // Graph inputs (pre-existing GPU tensors)
  Input,

  // Element-wise
  BinaryOp,
  UnaryOp,
  VecScalarOp,

  // Reduction
  Reduce,

  // Matrix
  MatMul,
  Transpose,
  Dot,

  // Special
  Clamp,
  Where,

  // Cumulative
  CumOp,

  // Statistical
  Variance,

  // Softmax
  Softmax,
  LogSoftmax,

  // Shape
  Reshape,
  Squeeze,
  Unsqueeze,
  Unflatten,
  Flatten,

  // Norm
  Norm,

  // Prefix scan
  PrefixScan,

  // Convolution
  Conv1d,
  Conv2d,

  // Pooling
  MaxPool2d,
  AvgPool2d,
  AdaptiveAvgPool2d,

  // Normalization
  LayerNorm,
  BatchNorm,

  // Embedding
  Embedding,

  // Padding
  Pad,
};

// ============================================================================
// NodeData structs — operation-specific parameters (no GPU state)
// ============================================================================

struct InputData {
  Tensor gpuHandle;
  bool isConstant = false;
};

struct BinaryOpData {
  OperatorEnum op;
};

struct UnaryOpData {
  OperatorEnum op;
};

struct VecScalarOpData {
  OperatorEnum op;
  float scalar;
};

struct ReduceData {
  OperatorEnum op;
  std::optional<int> dim;
};

struct MatMulData {};

struct TransposeData {};

struct DotData {};

struct ClampData {
  float minVal;
  float maxVal;
};

struct WhereData {};

struct CumOpData {
  OperatorEnum op;
  std::optional<int> dim;
};

struct VarianceData {
  int correction;
  std::optional<int> dim;
};

struct SoftmaxData {
  int dim;
};

struct ReshapeData {
  std::vector<int32_t> newShape;
};

struct SqueezeData {
  std::optional<int> dim;
};

struct UnsqueezeData {
  int dim;
};

struct UnflattenData {
  int dim;
  std::vector<uint32_t> sizes;
};

struct FlattenData {
  int startDim;
  int endDim;
};

struct NormData {
  std::optional<int> dim;
};

struct PrefixScanData {
  OperatorEnum op;
};

struct Conv1dData {
  uint32_t stride;
  uint32_t padding;
};

struct Conv2dData {
  uint32_t strideH;
  uint32_t strideW;
  uint32_t padH;
  uint32_t padW;
};

struct Pool2dData {
  uint32_t kernelH;
  uint32_t kernelW;
  uint32_t strideH;
  uint32_t strideW;
  uint32_t padH;
  uint32_t padW;
};

struct AdaptivePool2dData {
  uint32_t outH;
  uint32_t outW;
};

struct LayerNormData {
  std::vector<uint32_t> normalizedShape;
  float eps;
  bool hasWeight;
  bool hasBias;
};

struct BatchNormData {
  float eps;
  bool hasWeight;
  bool hasBias;
};

struct EmbeddingData {};

struct PadData {
  std::vector<uint32_t> padWidths;
  float value;
};

// ============================================================================
// NodeData variant
// ============================================================================

using NodeData = std::variant<InputData,
                              BinaryOpData,
                              UnaryOpData,
                              VecScalarOpData,
                              ReduceData,
                              MatMulData,
                              TransposeData,
                              DotData,
                              ClampData,
                              WhereData,
                              CumOpData,
                              VarianceData,
                              SoftmaxData,
                              ReshapeData,
                              SqueezeData,
                              UnsqueezeData,
                              UnflattenData,
                              FlattenData,
                              NormData,
                              PrefixScanData,
                              Conv1dData,
                              Conv2dData,
                              Pool2dData,
                              AdaptivePool2dData,
                              LayerNormData,
                              BatchNormData,
                              EmbeddingData,
                              PadData>;

// ============================================================================
// GraphNode — a single node in the computation graph
// ============================================================================

struct GraphNode {
  GraphNodeType type;

  /// Input VirtualTensors (0 for Input, 1 for unary/reshape, 2+ for
  /// binary/ternary).
  std::vector<VirtualTensor> inputs;

  /// Inferred output shape and dtype (computed during graph construction).
  std::vector<uint32_t> outputShape;
  DataType outputDtype = DataType::Float32;

  /// Operation-specific parameters.
  NodeData data;

  /// How many other nodes reference this node's output.
  uint32_t refCount = 0;

  /// Whether this node is a graph output (its result is returned to the
  /// caller).
  bool isOutput = false;

  /// Tombstone flag for dead nodes.
  bool isRemoved = false;
};

} // namespace graph
} // namespace cut
