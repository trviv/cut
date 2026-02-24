#include "Graph.h"
#include "OpNode.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace cut {
namespace graph {

uint32_t Graph::addNode(std::unique_ptr<OpNode> node,
                        const Tensor &outputTensor) {
  uint32_t idx = static_cast<uint32_t>(nodes_.size());

  // Increment graphRefCount on all input nodes
  for (uint32_t inputId : node->graphInputIds()) {
    if (inputId < nodes_.size() && nodes_[inputId]) {
      nodes_[inputId]->setGraphRefCount(nodes_[inputId]->graphRefCount() + 1);
    }
  }

  nodes_.push_back(std::move(node));
  tensorToNodeId_.emplace_back(outputTensor, idx);
  return idx;
}

uint32_t Graph::addNode(std::unique_ptr<OpNode> node) {
  uint32_t idx = static_cast<uint32_t>(nodes_.size());

  // Increment graphRefCount on all input nodes
  for (uint32_t inputId : node->graphInputIds()) {
    if (inputId < nodes_.size() && nodes_[inputId]) {
      nodes_[inputId]->setGraphRefCount(nodes_[inputId]->graphRefCount() + 1);
    }
  }

  nodes_.push_back(std::move(node));
  return idx;
}

OpNode &Graph::node(uint32_t id) {
  if (id == UINT32_MAX || id >= nodes_.size() || !nodes_[id]) {
    throw std::out_of_range("Invalid node id");
  }
  return *nodes_[id];
}

const OpNode &Graph::node(uint32_t id) const {
  if (id == UINT32_MAX || id >= nodes_.size() || !nodes_[id]) {
    throw std::out_of_range("Invalid node id");
  }
  return *nodes_[id];
}

OpNode &Graph::node(const Tensor &t) {
  return node(nodeId(t));
}

const OpNode &Graph::node(const Tensor &t) const {
  return node(nodeId(t));
}

uint32_t Graph::nodeId(const Tensor &t) const {
  for (const auto &p : tensorToNodeId_) {
    if (p.first == t)
      return p.second;
  }
  throw std::out_of_range("No node found for tensor");
}

size_t Graph::size() const {
  return nodes_.size();
}

std::vector<std::unique_ptr<OpNode>> &Graph::nodes() {
  return nodes_;
}

const std::vector<std::unique_ptr<OpNode>> &Graph::nodes() const {
  return nodes_;
}

void Graph::markOutput(uint32_t id) {
  if (id != UINT32_MAX && id < nodes_.size() && nodes_[id]) {
    nodes_[id]->setGraphOutput(true);
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

  // Replace in all node graphInputIds
  for (auto &n : nodes_) {
    if (!n || n->isGraphRemoved())
      continue;
    auto ids = n->graphInputIds();
    bool changed = false;
    for (auto &id : ids) {
      if (id == oldId) {
        id = newId;
        changed = true;
      }
    }
    if (changed)
      n->setGraphInputIds(std::move(ids));
  }

  // Replace in outputs list
  for (auto &out : outputs_) {
    if (out == oldId) {
      out = newId;
      // Transfer output marking
      if (newId != UINT32_MAX && newId < nodes_.size() && nodes_[newId]) {
        nodes_[newId]->setGraphOutput(true);
      }
    }
  }

  // Recompute refCounts (simpler and safer than incremental tracking)
  recomputeRefCounts();
}

void Graph::recomputeRefCounts() {
  // Reset all
  for (auto &n : nodes_) {
    if (n)
      n->setGraphRefCount(0);
  }

  // Count from node graphInputIds
  for (const auto &n : nodes_) {
    if (!n || n->isGraphRemoved())
      continue;
    for (uint32_t inputId : n->graphInputIds()) {
      if (inputId < nodes_.size() && nodes_[inputId]) {
        nodes_[inputId]->setGraphRefCount(nodes_[inputId]->graphRefCount() + 1);
      }
    }
  }

  // Count from output list
  for (const auto &out : outputs_) {
    if (out != UINT32_MAX && out < nodes_.size() && nodes_[out]) {
      nodes_[out]->setGraphRefCount(nodes_[out]->graphRefCount() + 1);
    }
  }
}

std::vector<uint32_t> Graph::topologicalOrder() const {
  size_t n = nodes_.size();

  // Compute in-degrees (only from non-removed nodes)
  std::vector<uint32_t> inDegree(n, 0);
  for (uint32_t i = 0; i < n; ++i) {
    if (!nodes_[i] || nodes_[i]->isGraphRemoved())
      continue;
    for (uint32_t inputId : nodes_[i]->graphInputIds()) {
      if (inputId < n && nodes_[inputId] &&
          !nodes_[inputId]->isGraphRemoved()) {
        inDegree[i]++;
      }
    }
  }

  // Start with nodes that have no inputs (in-degree 0)
  std::queue<uint32_t> ready;
  for (uint32_t i = 0; i < n; ++i) {
    if (nodes_[i] && !nodes_[i]->isGraphRemoved() && inDegree[i] == 0) {
      ready.push(i);
    }
  }

  // BFS topological sort
  std::vector<uint32_t> order;
  order.reserve(n);

  // Build adjacency: for each node, which nodes depend on it
  std::vector<std::vector<uint32_t>> dependents(n);
  for (uint32_t i = 0; i < n; ++i) {
    if (!nodes_[i] || nodes_[i]->isGraphRemoved())
      continue;
    for (uint32_t inputId : nodes_[i]->graphInputIds()) {
      if (inputId < n && nodes_[inputId] &&
          !nodes_[inputId]->isGraphRemoved()) {
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
    if (!n) {
      copy.nodes_.push_back(nullptr);
      continue;
    }

    // Determine detail string for reporting
    std::string detail;
    if (n->isInputNode()) {
      auto *inp = dynamic_cast<const InputOpNode *>(n.get());
      detail = (inp && inp->isConstant()) ? "constant" : "dynamic";
    }

    auto stub = std::make_unique<StubOpNode>(
        n->op(), n->outputShape(), n->outputDtype(), n->displayName(), detail);

    // Copy graph metadata
    stub->setGraphInputIds(std::vector<uint32_t>(n->graphInputIds().begin(),
                                                 n->graphInputIds().end()));
    stub->setGraphRefCount(n->graphRefCount());
    stub->setGraphOutput(n->isGraphOutput());
    stub->setGraphRemoved(n->isGraphRemoved());

    // Preserve logical type and input status
    stub->setLogicalType(n->logicalType());
    stub->setIsInput(n->isInputNode());

    copy.nodes_.push_back(std::move(stub));
  }
  copy.outputs_ = outputs_;
  return copy;
}

} // namespace graph
} // namespace cut
