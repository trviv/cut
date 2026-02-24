#pragma once

#include "GraphNode.h"
#include "OpNode.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace cut {

namespace graph {

/// Container for a computation DAG. Owns all nodes (as OpNode pointers) and
/// provides graph manipulation primitives used by optimization passes.
class Graph {
public:
  Graph() = default;
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph(Graph &&) = default;
  Graph &operator=(Graph &&) = default;

  /// Adds a node to the graph. Increments graphRefCount on all input nodes
  /// (identified by the node's graphInputIds). Returns a VirtualTensor
  /// referencing the new node's output.
  VirtualTensor addNode(std::unique_ptr<OpNode> node);

  /// Returns a mutable reference to a node.
  OpNode &node(VirtualTensor vt);
  const OpNode &node(VirtualTensor vt) const;

  /// Returns the number of nodes (including tombstones).
  size_t size() const;

  /// Direct access to the node vector for iteration by passes.
  std::vector<std::unique_ptr<OpNode>> &nodes();
  const std::vector<std::unique_ptr<OpNode>> &nodes() const;

  /// Mark a VirtualTensor as a graph output.
  void markOutput(VirtualTensor vt);

  /// Returns the list of output VirtualTensors.
  const std::vector<VirtualTensor> &outputs() const;

  /// Replace all uses of oldVt with newVt throughout the graph.
  /// Updates graphRefCounts on both old and new nodes. Also updates outputs
  /// list.
  void replaceAllUses(VirtualTensor oldVt, VirtualTensor newVt);

  /// Recompute all graphRefCounts from scratch by scanning every node's
  /// graphInputIds and the output list.
  void recomputeRefCounts();

  /// Returns node indices in topological (dependency) order.
  /// Skips removed nodes. Uses Kahn's algorithm.
  std::vector<uint32_t> topologicalOrder() const;

  /// Deep-copy the graph structure using StubOpNodes.
  /// Suitable for snapshots used in reporting/visualization but not execution.
  Graph clone() const;

private:
  std::vector<std::unique_ptr<OpNode>> nodes_;
  std::vector<VirtualTensor> outputs_;
};

} // namespace graph
} // namespace cut
