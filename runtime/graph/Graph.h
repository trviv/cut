#pragma once

#include "GraphNode.h"

#include <cstdint>
#include <vector>

namespace cut {
namespace graph {

/// Container for a computation DAG. Owns all nodes and provides graph
/// manipulation primitives used by optimization passes.
class Graph {
public:
  /// Adds a node to the graph. Increments refCount on all input nodes.
  /// Returns a VirtualTensor referencing the new node's output.
  VirtualTensor addNode(GraphNode &&node);

  /// Returns a mutable reference to a node.
  GraphNode &node(VirtualTensor vt);
  const GraphNode &node(VirtualTensor vt) const;

  /// Returns the number of nodes (including tombstones).
  size_t size() const;

  /// Direct access to the node vector for iteration by passes.
  std::vector<GraphNode> &nodes();
  const std::vector<GraphNode> &nodes() const;

  /// Mark a VirtualTensor as a graph output.
  void markOutput(VirtualTensor vt);

  /// Returns the list of output VirtualTensors.
  const std::vector<VirtualTensor> &outputs() const;

  /// Replace all uses of oldVt with newVt throughout the graph.
  /// Updates refCounts on both old and new nodes. Also updates outputs list.
  void replaceAllUses(VirtualTensor oldVt, VirtualTensor newVt);

  /// Recompute all refCounts from scratch by scanning every node's inputs
  /// and the output list.
  void recomputeRefCounts();

  /// Returns node indices in topological (dependency) order.
  /// Skips removed nodes. Uses Kahn's algorithm.
  std::vector<uint32_t> topologicalOrder() const;

private:
  std::vector<GraphNode> nodes_;
  std::vector<VirtualTensor> outputs_;
};

} // namespace graph
} // namespace cut
