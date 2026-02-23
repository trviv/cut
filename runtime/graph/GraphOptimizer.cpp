#include "GraphOptimizer.h"

#include <algorithm>

namespace cut {
namespace graph {

// ============================================================================
// GraphOptimizer
// ============================================================================

void GraphOptimizer::addPass(std::unique_ptr<GraphPass> pass) {
  passes_.push_back(std::move(pass));
}

void GraphOptimizer::optimize(Graph &graph) {
  // Run passes to fixed-point: repeat until no pass modifies the graph
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &pass : passes_) {
      if (pass->run(graph)) {
        changed = true;
      }
    }
  }
}

GraphOptimizer GraphOptimizer::createDefault() {
  GraphOptimizer opt;
  opt.addPass(std::make_unique<IdentityReshapePass>());
  opt.addPass(std::make_unique<ReshapeChainPass>());
  opt.addPass(std::make_unique<TransposeCancelPass>());
  opt.addPass(std::make_unique<DeadCodePass>());
  return opt;
}

// ============================================================================
// IdentityReshapePass
// ============================================================================

static bool isReshapeLike(GraphNodeType type) {
  return type == GraphNodeType::Reshape || type == GraphNodeType::Squeeze ||
         type == GraphNodeType::Unsqueeze || type == GraphNodeType::Flatten ||
         type == GraphNodeType::Unflatten;
}

bool IdentityReshapePass::run(Graph &graph) {
  bool changed = false;

  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &n = graph.nodes()[i];
    if (n.isRemoved || !isReshapeLike(n.type))
      continue;
    if (n.inputs.empty())
      continue;

    const auto &inputNode = graph.node(n.inputs[0]);
    if (n.outputShape == inputNode.outputShape) {
      // This reshape is a no-op — rewire consumers to use the input directly
      graph.replaceAllUses(VirtualTensor{i}, n.inputs[0]);
      changed = true;
    }
  }

  return changed;
}

// ============================================================================
// ReshapeChainPass
// ============================================================================

bool ReshapeChainPass::run(Graph &graph) {
  bool changed = false;

  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &n = graph.nodes()[i];
    if (n.isRemoved || n.type != GraphNodeType::Reshape)
      continue;
    if (n.inputs.empty())
      continue;

    auto &inputNode = graph.node(n.inputs[0]);
    if (inputNode.isRemoved || inputNode.type != GraphNodeType::Reshape)
      continue;

    // Skip the intermediate reshape: point this node at the inner input
    n.inputs[0] = inputNode.inputs[0];
    graph.recomputeRefCounts();
    changed = true;
  }

  return changed;
}

// ============================================================================
// TransposeCancelPass
// ============================================================================

bool TransposeCancelPass::run(Graph &graph) {
  bool changed = false;

  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &n = graph.nodes()[i];
    if (n.isRemoved || n.type != GraphNodeType::Transpose)
      continue;
    if (n.inputs.empty())
      continue;

    const auto &inputNode = graph.node(n.inputs[0]);
    if (inputNode.isRemoved || inputNode.type != GraphNodeType::Transpose)
      continue;

    // transpose(transpose(x)) → x
    graph.replaceAllUses(VirtualTensor{i}, inputNode.inputs[0]);
    changed = true;
  }

  return changed;
}

// ============================================================================
// DeadCodePass
// ============================================================================

bool DeadCodePass::run(Graph &graph) {
  bool changed = false;

  // Recompute refCounts to be safe
  graph.recomputeRefCounts();

  // Walk in reverse topological order so that removing a node may cascade
  auto order = graph.topologicalOrder();
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    uint32_t idx = *it;
    auto &n = graph.nodes()[idx];
    if (n.isRemoved)
      continue;

    // Don't remove outputs or inputs
    if (n.isOutput)
      continue;
    if (n.type == GraphNodeType::Input)
      continue;

    if (n.refCount == 0) {
      // Decrement refCount on this node's inputs
      for (const auto &input : n.inputs) {
        if (input.isValid() && input.id < graph.size()) {
          auto &inputNode = graph.nodes()[input.id];
          if (inputNode.refCount > 0)
            inputNode.refCount--;
        }
      }
      n.isRemoved = true;
      changed = true;
    }
  }

  return changed;
}

} // namespace graph
} // namespace cut
