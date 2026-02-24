#pragma once

#include "GraphNode.h"
#include "OpNode.h"

#include <ComputeHandle.h>

#include <cstdint>
#include <memory>
#include <utility>
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
  /// (identified by the node's graphInputIds). Returns the node index.
  /// @param outputTensor The Tensor associated with this node's output
  ///        (used for Tensor-based lookups).
  uint32_t addNode(std::unique_ptr<OpNode> node, const Tensor &outputTensor);

  /// Adds a node without an associated output tensor (for internal use).
  uint32_t addNode(std::unique_ptr<OpNode> node);

  /// Returns a mutable reference to a node by index.
  OpNode &node(uint32_t id);
  const OpNode &node(uint32_t id) const;

  /// Returns a mutable reference to a node by its associated Tensor.
  OpNode &node(const Tensor &t);
  const OpNode &node(const Tensor &t) const;

  /// Returns the node index for a given Tensor.
  uint32_t nodeId(const Tensor &t) const;

  /// Returns the number of nodes (including tombstones).
  size_t size() const;

  /// Direct access to the node vector for iteration by passes.
  std::vector<std::unique_ptr<OpNode>> &nodes();
  const std::vector<std::unique_ptr<OpNode>> &nodes() const;

  /// Mark a node (by index) as a graph output.
  void markOutput(uint32_t id);

  /// Mark a node (by Tensor) as a graph output.
  void markOutput(const Tensor &t);

  /// Returns the list of output node indices.
  const std::vector<uint32_t> &outputs() const;

  /// Replace all uses of oldId with newId throughout the graph.
  /// Updates graphRefCounts on both old and new nodes. Also updates outputs
  /// list.
  void replaceAllUses(uint32_t oldId, uint32_t newId);

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
  std::vector<uint32_t> outputs_;

  /// Maps Tensor handle → node index for Tensor-based lookups.
  std::vector<std::pair<Tensor, uint32_t>> tensorToNodeId_;
};

} // namespace graph
} // namespace cut
