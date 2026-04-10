#pragma once

#include "Graph.h"

#include <ComputeHandle.h>

#include <unordered_map>
#include <vector>

namespace cut {

class Operations;
class TensorStore;

namespace graph {

/// Executes an optimized computation graph by walking it in topological order
/// and dispatching each node through the existing Operations API.
/// Runs MemoryPlanner before execution to minimize transient tensor memory.
class GraphExecutor {
public:
  GraphExecutor(Operations &ops, TensorStore &store);

  /// Execute the graph and return GPU Tensor handles for each marked output.
  /// The returned vector has one Tensor per graph output, in the same order
  /// as markOutput() calls during graph construction.
  std::vector<Tensor> execute(Graph &graph);

private:
  Operations *ops_;
  TensorStore *store_;

  /// Maps node index → real GPU Tensor during execution.
  std::vector<Tensor> tensorMap_;

  /// Cached topological order per graph (keyed by graph pointer).
  /// Avoids recomputing topological order and replanning memory on every
  /// execution of the same immutable graph template.
  struct CachedPlan {
    std::vector<uint32_t> order;
    bool planned = false;
  };
  std::unordered_map<const Graph *, CachedPlan> cachedPlans_;

  /// Execute a single graph node, populating tensorMap_[nodeIndex].
  void executeNode(Graph &graph, uint32_t nodeIndex);
};

} // namespace graph
} // namespace cut
