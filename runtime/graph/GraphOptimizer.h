#pragma once

#include "Graph.h"

#include <memory>
#include <string>
#include <vector>

namespace cut {

class TensorStore; // Forward declaration

namespace graph {

/// Abstract base for graph optimization passes.
class GraphPass {
public:
  virtual ~GraphPass() = default;

  /// Human-readable name for debugging/logging.
  virtual const char *name() const = 0;

  /// Apply this pass to the graph. Returns true if the graph was modified.
  /// @param graph The computation graph to optimize
  /// @param store TensorStore for creating new tensors (used by fusion passes)
  virtual bool run(Graph &graph, TensorStore &store) = 0;
};

/// Statistics for a single optimization pass.
struct PassStats {
  std::string name;
  int runCount = 0; // Number of times this pass modified the graph
};

/// Runs a pipeline of optimization passes on a computation graph.
class GraphOptimizer {
public:
  /// Add a pass to the pipeline.
  void addPass(std::unique_ptr<GraphPass> pass);

  /// Run all passes. Repeats the full pipeline until no pass reports a change
  /// (fixed-point iteration).
  /// @param graph The computation graph to optimize
  /// @param store TensorStore for creating new tensors (used by fusion passes)
  void optimize(Graph &graph, TensorStore &store);

  /// Create an optimizer with the default set of passes.
  static GraphOptimizer createDefault();

  /// Get statistics from the last optimize() call.
  const std::vector<PassStats> &stats() const { return stats_; }

  /// Reset statistics.
  void resetStats() { stats_.clear(); }

private:
  std::vector<std::unique_ptr<GraphPass>> passes_;
  std::vector<PassStats> stats_;
};

// ============================================================================
// Built-in optimization passes
// ============================================================================

/// Eliminates reshape-like nodes (Reshape, Squeeze, Unsqueeze, Flatten,
/// Unflatten) whose output shape is identical to their input shape.
class IdentityReshapePass : public GraphPass {
public:
  const char *name() const override { return "IdentityReshape"; }
  bool run(Graph &graph, TensorStore &store) override;
};

/// Collapses chains of Reshape nodes: reshape(reshape(x, s1), s2) → reshape(x,
/// s2). The intermediate reshape becomes dead and is cleaned up by DCE.
class ReshapeChainPass : public GraphPass {
public:
  const char *name() const override { return "ReshapeChain"; }
  bool run(Graph &graph, TensorStore &store) override;
};

/// Cancels consecutive transpose pairs: transpose(transpose(x)) → x.
class TransposeCancelPass : public GraphPass {
public:
  const char *name() const override { return "TransposeCancel"; }
  bool run(Graph &graph, TensorStore &store) override;
};

/// Eliminates reshape nodes whose memory layout is identical to their input.
/// Covers cases where shapes differ (including dimensionality changes like
/// [576] → [1,576]) but innermost dimensions share the same alignment,
/// making the GPU copy a no-op.
class NoOpReshapePass : public GraphPass {
public:
  const char *name() const override { return "NoOpReshape"; }
  bool run(Graph &graph, TensorStore &store) override;
};

/// Fuses VecVecAdd → UnarySquare → ReduceSum → VecScalarMul → VecVecMul into
/// ExtendedRMSNorm (residual + normalization in single kernel).
class ExtendedRMSNormFusionPass : public GraphPass {
public:
  const char *name() const override { return "ExtendedRMSNormFusion"; }
  bool run(Graph &graph, TensorStore &store) override;
};

/// Fuses UnarySquare → ReduceSum → VecScalarMul → VecVecMul into RMSNorm
/// (standard normalization in single kernel).
class RMSNormFusionPass : public GraphPass {
public:
  const char *name() const override { return "RMSNormFusion"; }
  bool run(Graph &graph, TensorStore &store) override;
};

/// Fuses MatMul → UnarySilu into MatMulSiLU (matmul with inline SiLU
/// activation). Used in FFN gate projections (30× per forward pass in SmolLM2).
class MatMulSiLUFusionPass : public GraphPass {
public:
  const char *name() const override { return "MatMulSiLUFusion"; }
  bool run(Graph &graph, TensorStore &store) override;
};

/// Fuses MatMul/MatMulQ8/MatMulQ4 → BinaryVecVec into MatMul with Binary
/// fusion (matmul with inline binary operation via SPIR-V linking).
class MatMulBinaryFusionPass : public GraphPass {
public:
  const char *name() const override { return "MatMulBinaryFusion"; }
  bool run(Graph &graph, TensorStore &store) override;
};

/// Fuses two consecutive binary operations into a single dispatch.
/// Supports VecScalar→VecVec and VecVec→VecScalar patterns.
class FusedBinaryPass : public GraphPass {
public:
  const char *name() const override { return "FusedBinary"; }
  bool run(Graph &graph, TensorStore &store) override;
};

/// Removes nodes with refCount == 0 that are not graph outputs.
/// Runs in reverse topological order so removals cascade.
class DeadCodePass : public GraphPass {
public:
  const char *name() const override { return "DeadCode"; }
  bool run(Graph &graph, TensorStore &store) override;
};

} // namespace graph
} // namespace cut
