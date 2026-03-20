#include "Graph.h"
#include "OpNode.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace cut {
namespace graph {

// Populate cached metadata from the OpNode into the GraphNode.
static void populateMetadata(GraphNode &gn) {
  gn.logicalType = gn.op->logicalType();
  gn.isInput = gn.op->isInputNode();
  gn.displayName = gn.op->displayName();
  gn.outputShape = gn.op->outputShape();
  gn.outputDtype = gn.op->outputDtype();
  // Detail for input nodes (constant vs dynamic)
  if (gn.isInput) {
    auto *inp = dynamic_cast<const InputOpNode *>(gn.op.get());
    gn.detail = (inp && inp->isConstant()) ? "constant" : "dynamic";
  }
}

uint32_t Graph::addNode(std::unique_ptr<OpNode> node,
                        const Tensor &outputTensor,
                        std::vector<uint32_t> inputIds) {
  uint32_t idx = static_cast<uint32_t>(nodes_.size());

  // Increment refCount on all input nodes
  for (uint32_t inputId : inputIds) {
    if (inputId < nodes_.size() && nodes_[inputId].op) {
      nodes_[inputId].refCount++;
    }
  }

  GraphNode gn;
  gn.op = std::move(node);
  gn.inputIds = std::move(inputIds);
  populateMetadata(gn);
  nodes_.push_back(std::move(gn));
  tensorToNodeId_[outputTensor.id()] = idx;
  return idx;
}

uint32_t Graph::addNode(std::unique_ptr<OpNode> node,
                        std::vector<uint32_t> inputIds) {
  uint32_t idx = static_cast<uint32_t>(nodes_.size());

  // Increment refCount on all input nodes
  for (uint32_t inputId : inputIds) {
    if (inputId < nodes_.size() && nodes_[inputId].op) {
      nodes_[inputId].refCount++;
    }
  }

  GraphNode gn;
  gn.op = std::move(node);
  gn.inputIds = std::move(inputIds);
  populateMetadata(gn);
  nodes_.push_back(std::move(gn));
  return idx;
}

GraphNode &Graph::node(uint32_t id) {
  if (id == UINT32_MAX || id >= nodes_.size() || !nodes_[id].op) {
    throw std::out_of_range("Invalid node id");
  }
  return nodes_[id];
}

const GraphNode &Graph::node(uint32_t id) const {
  if (id == UINT32_MAX || id >= nodes_.size() || !nodes_[id].op) {
    throw std::out_of_range("Invalid node id");
  }
  return nodes_[id];
}

GraphNode &Graph::node(const Tensor &t) {
  return node(nodeId(t));
}

const GraphNode &Graph::node(const Tensor &t) const {
  return node(nodeId(t));
}

uint32_t Graph::nodeId(const Tensor &t) const {
  auto it = tensorToNodeId_.find(t.id());
  if (it != tensorToNodeId_.end())
    return it->second;
  throw std::out_of_range("No node found for tensor");
}

std::optional<uint32_t> Graph::tryNodeId(const Tensor &t) const {
  auto it = tensorToNodeId_.find(t.id());
  if (it != tensorToNodeId_.end())
    return it->second;
  return std::nullopt;
}

size_t Graph::size() const {
  return nodes_.size();
}

std::vector<GraphNode> &Graph::nodes() {
  return nodes_;
}

const std::vector<GraphNode> &Graph::nodes() const {
  return nodes_;
}

void Graph::markOutput(uint32_t id) {
  if (id != UINT32_MAX && id < nodes_.size() && nodes_[id].op) {
    nodes_[id].isOutput = true;
    outputs_.push_back(id);
  }
}

void Graph::markOutput(const Tensor &t) {
  markOutput(nodeId(t));
}

const std::vector<uint32_t> &Graph::outputs() const {
  return outputs_;
}

void Graph::replaceAllUses(uint32_t oldId, uint32_t newId) {
  if (oldId == newId)
    return;

  // Replace in all node inputIds
  for (auto &n : nodes_) {
    if (!n.op || n.isRemoved)
      continue;
    for (auto &id : n.inputIds) {
      if (id == oldId) {
        id = newId;
      }
    }
  }

  // Replace in outputs list
  for (auto &out : outputs_) {
    if (out == oldId) {
      out = newId;
      // Transfer output marking
      if (newId != UINT32_MAX && newId < nodes_.size() && nodes_[newId].op) {
        nodes_[newId].isOutput = true;
      }
    }
  }

  // Clear isOutput on old node if no output entry still references it
  if (oldId < nodes_.size() && nodes_[oldId].isOutput) {
    bool stillOutput = false;
    for (auto out : outputs_) {
      if (out == oldId) {
        stillOutput = true;
        break;
      }
    }
    nodes_[oldId].isOutput = stillOutput;
  }

  // Recompute refCounts (simpler and safer than incremental tracking)
  recomputeRefCounts();
}

void Graph::recomputeRefCounts() {
  // Reset all
  for (auto &n : nodes_) {
    if (n.op)
      n.refCount = 0;
  }

  // Count from node inputIds
  for (const auto &n : nodes_) {
    if (!n.op || n.isRemoved)
      continue;
    for (uint32_t inputId : n.inputIds) {
      if (inputId < nodes_.size() && nodes_[inputId].op) {
        nodes_[inputId].refCount++;
      }
    }
  }

  // Count from output list
  for (const auto &out : outputs_) {
    if (out != UINT32_MAX && out < nodes_.size() && nodes_[out].op) {
      nodes_[out].refCount++;
    }
  }
}

std::vector<uint32_t> Graph::topologicalOrder() const {
  size_t n = nodes_.size();

  // A node is valid if it's not removed and either has an OpNode (live graph)
  // or has metadata (cloned graph for reporting).
  auto isValid = [&](uint32_t i) -> bool {
    return !nodes_[i].isRemoved &&
           (nodes_[i].op || !nodes_[i].displayName.empty());
  };

  // Compute in-degrees (only from valid nodes)
  std::vector<uint32_t> inDegree(n, 0);
  for (uint32_t i = 0; i < n; ++i) {
    if (!isValid(i))
      continue;
    for (uint32_t inputId : nodes_[i].inputIds) {
      if (inputId < n && isValid(inputId)) {
        inDegree[i]++;
      }
    }
  }

  // Start with nodes that have no inputs (in-degree 0)
  std::queue<uint32_t> ready;
  for (uint32_t i = 0; i < n; ++i) {
    if (isValid(i) && inDegree[i] == 0) {
      ready.push(i);
    }
  }

  // BFS topological sort
  std::vector<uint32_t> order;
  order.reserve(n);

  // Build adjacency: for each node, which nodes depend on it
  std::vector<std::vector<uint32_t>> dependents(n);
  for (uint32_t i = 0; i < n; ++i) {
    if (!isValid(i))
      continue;
    for (uint32_t inputId : nodes_[i].inputIds) {
      if (inputId < n && isValid(inputId)) {
        dependents[inputId].push_back(i);
      }
    }
  }

  while (!ready.empty()) {
    uint32_t curr = ready.front();
    ready.pop();
    order.push_back(curr);

    for (uint32_t dep : dependents[curr]) {
      if (--inDegree[dep] == 0) {
        ready.push(dep);
      }
    }
  }

  return order;
}

Graph Graph::clone() const {
  Graph copy;
  copy.nodes_.reserve(nodes_.size());
  for (const auto &n : nodes_) {
    GraphNode gn;
    if (!n.op) {
      // Empty slot — mark as removed so topology traversal skips it.
      gn.isRemoved = true;
      copy.nodes_.push_back(std::move(gn));
      continue;
    }
    // Copy all metadata — no OpNode needed for reporting/visualization.
    gn.inputIds = n.inputIds;
    gn.refCount = n.refCount;
    gn.isOutput = n.isOutput;
    gn.isRemoved = n.isRemoved;
    gn.logicalType = n.logicalType;
    gn.isInput = n.isInput;
    gn.displayName = n.displayName;
    gn.outputShape = n.outputShape;
    gn.outputDtype = n.outputDtype;
    gn.detail = n.detail;
    copy.nodes_.push_back(std::move(gn));
  }
  copy.outputs_ = outputs_;
  return copy;
}

} // namespace graph
} // namespace cut
