#pragma once

#include "Graph.h"

#include <ComputeHandle.h>

#include <vector>

namespace cut {

class Operations;
class Runtime;

namespace graph {

/// Executes an optimized computation graph by walking it in topological order
/// and dispatching each node through the existing Operations API or directly
/// encoding stored OpNodes.
class GraphExecutor {
public:
  GraphExecutor(Operations &ops, Runtime &runtime);

  /// Execute the graph and return GPU Tensor handles for each marked output.
  /// The returned vector has one Tensor per graph output, in the same order
  /// as markOutput() calls during graph construction.
  /// OpNodes stored in the graph are consumed (moved) during execution.
  std::vector<Tensor> execute(Graph &graph);

private:
  Operations *ops_;
  Runtime *runtime_;

  /// Maps node index → real GPU Tensor during execution.
  std::vector<Tensor> tensorMap_;

  /// Execute a single graph node, populating tensorMap_[nodeIndex].
  void executeNode(Graph &graph, uint32_t nodeIndex);
};

} // namespace graph
} // namespace cut
