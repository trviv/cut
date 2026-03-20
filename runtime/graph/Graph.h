#pragma once

#include "OpNode.h"

#include <ComputeCommon.h>
#include <ComputeHandle.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cut {

namespace graph {

/// Wraps an OpNode with graph-level topology and metadata.
/// OpNode handles GPU dispatch; GraphNode handles DAG structure.
/// Metadata fields (outputShape, outputDtype, detail) are cached at addNode()
/// time so that optimizer passes and reporting can access them without reaching
/// through to the OpNode.
struct GraphNode {
  std::unique_ptr<OpNode> op;
  std::vector<uint32_t> inputIds;
  uint32_t refCount = 0;
  bool isOutput = false;
  bool isRemoved = false;
  LogicalOpType logicalType = LogicalOpType::Other;
  bool isInput = false;
  std::string displayName;
  // Cached metadata (populated at addNode, used by clone/report/optimizer):
  std::vector<uint32_t> outputShape;
  DataType outputDtype = DataType::Float32;
  std::string detail; // e.g. "constant"/"dynamic" for inputs
};

/// Container for a computation DAG. Owns all nodes (as GraphNode wrappers) and
/// provides graph manipulation primitives used by optimization passes.
class Graph {
public:
  Graph() = default;
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph(Graph &&) = default;
  Graph &operator=(Graph &&) = default;

  /// Adds a node to the graph. Increments refCount on all input nodes.
  /// Returns the node index.
  /// @param outputTensor The Tensor associated with this node's output
  ///        (used for Tensor-based lookups).
  /// @param inputIds Node indices of this node's inputs (DAG edges).
  uint32_t addNode(std::unique_ptr<OpNode> node,
                   const Tensor &outputTensor,
                   std::vector<uint32_t> inputIds = {});

  /// Adds a node without an associated output tensor (for internal use).
  uint32_t addNode(std::unique_ptr<OpNode> node,
                   std::vector<uint32_t> inputIds = {});

  /// Returns a mutable reference to a GraphNode by index.
  GraphNode &node(uint32_t id);
  const GraphNode &node(uint32_t id) const;

  /// Returns a mutable reference to a GraphNode by its associated Tensor.
  GraphNode &node(const Tensor &t);
  const GraphNode &node(const Tensor &t) const;

  /// Returns the node index for a given Tensor.
  uint32_t nodeId(const Tensor &t) const;

  /// Returns the node index for a given Tensor, or nullopt if not found.
  std::optional<uint32_t> tryNodeId(const Tensor &t) const;

  /// Returns the number of nodes (including tombstones).
  size_t size() const;

  /// Direct access to the node vector for iteration by passes.
  std::vector<GraphNode> &nodes();
  const std::vector<GraphNode> &nodes() const;

  /// Mark a node (by index) as a graph output.
  void markOutput(uint32_t id);

  /// Mark a node (by Tensor) as a graph output.
  void markOutput(const Tensor &t);

  /// Returns the list of output node indices.
  const std::vector<uint32_t> &outputs() const;

  /// Replace all uses of oldId with newId throughout the graph.
  /// Updates refCounts on both old and new nodes. Also updates outputs list.
  void replaceAllUses(uint32_t oldId, uint32_t newId);

  /// Recompute all refCounts from scratch by scanning every node's
  /// inputIds and the output list.
  void recomputeRefCounts();

  /// Returns node indices in topological (dependency) order.
  /// Skips removed nodes. Uses Kahn's algorithm.
  std::vector<uint32_t> topologicalOrder() const;

  /// Deep-copy the graph metadata (no OpNodes).
  /// Suitable for snapshots used in reporting/visualization but not execution.
  Graph clone() const;

  /// Mark the graph as executed. flush() checks this to avoid re-execution.
  void markExecuted() { executed_ = true; }
  bool isExecuted() const { return executed_; }

private:
  std::vector<GraphNode> nodes_;
  std::vector<uint32_t> outputs_;
  bool executed_ = false;

  /// Maps Tensor handle id → node index for O(1) Tensor-based lookups.
  std::unordered_map<size_t, uint32_t> tensorToNodeId_;
};

} // namespace graph
} // namespace cut
