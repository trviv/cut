#pragma once

#include "Graph.h"

#include <memory>
#include <vector>

namespace cut {
namespace graph {

/// Abstract base for graph optimization passes.
class GraphPass {
public:
  virtual ~GraphPass() = default;

  /// Human-readable name for debugging/logging.
  virtual const char *name() const = 0;

  /// Apply this pass to the graph. Returns true if the graph was modified.
  virtual bool run(Graph &graph) = 0;
};

/// Runs a pipeline of optimization passes on a computation graph.
class GraphOptimizer {
public:
  /// Add a pass to the pipeline.
  void addPass(std::unique_ptr<GraphPass> pass);

  /// Run all passes. Repeats the full pipeline until no pass reports a change
  /// (fixed-point iteration).
  void optimize(Graph &graph);

  /// Create an optimizer with the default set of passes.
  static GraphOptimizer createDefault();

private:
  std::vector<std::unique_ptr<GraphPass>> passes_;
};

// ============================================================================
// Built-in optimization passes
// ============================================================================

/// Eliminates reshape-like nodes (Reshape, Squeeze, Unsqueeze, Flatten,
/// Unflatten) whose output shape is identical to their input shape.
class IdentityReshapePass : public GraphPass {
public:
  const char *name() const override { return "IdentityReshape"; }
  bool run(Graph &graph) override;
};

/// Collapses chains of Reshape nodes: reshape(reshape(x, s1), s2) → reshape(x,
/// s2). The intermediate reshape becomes dead and is cleaned up by DCE.
class ReshapeChainPass : public GraphPass {
public:
  const char *name() const override { return "ReshapeChain"; }
  bool run(Graph &graph) override;
};

/// Cancels consecutive transpose pairs: transpose(transpose(x)) → x.
class TransposeCancelPass : public GraphPass {
public:
  const char *name() const override { return "TransposeCancel"; }
  bool run(Graph &graph) override;
};

/// Eliminates reshape nodes whose memory layout is identical to their input.
/// Covers cases where shapes differ but innermost dimensions share the same
/// alignment (e.g. [576] → [1,576]), making the GPU copy a no-op.
class NoOpReshapePass : public GraphPass {
public:
  const char *name() const override { return "NoOpReshape"; }
  bool run(Graph &graph) override;
};

/// Removes nodes with refCount == 0 that are not graph outputs.
/// Runs in reverse topological order so removals cascade.
class DeadCodePass : public GraphPass {
public:
  const char *name() const override { return "DeadCode"; }
  bool run(Graph &graph) override;
};

} // namespace graph
} // namespace cut
