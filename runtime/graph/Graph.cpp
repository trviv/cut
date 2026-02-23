#include "Graph.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace cut {
namespace graph {

VirtualTensor Graph::addNode(GraphNode &&node) {
  uint32_t idx = static_cast<uint32_t>(nodes_.size());

  // Increment refCount on all input nodes
  for (const auto &input : node.inputs) {
    if (input.isValid() && input.id < nodes_.size()) {
      nodes_[input.id].refCount++;
    }
  }

  nodes_.push_back(std::move(node));
  return VirtualTensor{idx};
}

GraphNode &Graph::node(VirtualTensor vt) {
  if (!vt.isValid() || vt.id >= nodes_.size()) {
    throw std::out_of_range("Invalid VirtualTensor id");
  }
  return nodes_[vt.id];
}

const GraphNode &Graph::node(VirtualTensor vt) const {
  if (!vt.isValid() || vt.id >= nodes_.size()) {
    throw std::out_of_range("Invalid VirtualTensor id");
  }
  return nodes_[vt.id];
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

void Graph::markOutput(VirtualTensor vt) {
  if (vt.isValid() && vt.id < nodes_.size()) {
    nodes_[vt.id].isOutput = true;
    outputs_.push_back(vt);
  }
}

const std::vector<VirtualTensor> &Graph::outputs() const {
  return outputs_;
}

void Graph::replaceAllUses(VirtualTensor oldVt, VirtualTensor newVt) {
  if (oldVt == newVt)
    return;

  // Replace in all node inputs
  for (auto &n : nodes_) {
    if (n.isRemoved)
      continue;
    for (auto &input : n.inputs) {
      if (input == oldVt) {
        input = newVt;
      }
    }
  }

  // Replace in outputs list
  for (auto &out : outputs_) {
    if (out == oldVt) {
      out = newVt;
      // Transfer output marking
      if (newVt.isValid() && newVt.id < nodes_.size()) {
        nodes_[newVt.id].isOutput = true;
      }
    }
  }

  // Recompute refCounts (simpler and safer than incremental tracking)
  recomputeRefCounts();
}

void Graph::recomputeRefCounts() {
  // Reset all
  for (auto &n : nodes_) {
    n.refCount = 0;
  }

  // Count from node inputs
  for (const auto &n : nodes_) {
    if (n.isRemoved)
      continue;
    for (const auto &input : n.inputs) {
      if (input.isValid() && input.id < nodes_.size()) {
        nodes_[input.id].refCount++;
      }
    }
  }

  // Count from output list
  for (const auto &out : outputs_) {
    if (out.isValid() && out.id < nodes_.size()) {
      nodes_[out.id].refCount++;
    }
  }
}

std::vector<uint32_t> Graph::topologicalOrder() const {
  size_t n = nodes_.size();

  // Compute in-degrees (only from non-removed nodes)
  std::vector<uint32_t> inDegree(n, 0);
  for (uint32_t i = 0; i < n; ++i) {
    if (nodes_[i].isRemoved)
      continue;
    for (const auto &input : nodes_[i].inputs) {
      if (input.isValid() && input.id < n && !nodes_[input.id].isRemoved) {
        inDegree[i]++;
      }
    }
  }

  // Start with nodes that have no inputs (in-degree 0)
  std::queue<uint32_t> ready;
  for (uint32_t i = 0; i < n; ++i) {
    if (!nodes_[i].isRemoved && inDegree[i] == 0) {
      ready.push(i);
    }
  }

  // BFS topological sort
  std::vector<uint32_t> order;
  order.reserve(n);

  // Build adjacency: for each node, which nodes depend on it
  std::vector<std::vector<uint32_t>> dependents(n);
  for (uint32_t i = 0; i < n; ++i) {
    if (nodes_[i].isRemoved)
      continue;
    for (const auto &input : nodes_[i].inputs) {
      if (input.isValid() && input.id < n && !nodes_[input.id].isRemoved) {
        dependents[input.id].push_back(i);
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

} // namespace graph
} // namespace cut
