#include "GraphOptimizer.h"
#include "OpNode.h"

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

bool IdentityReshapePass::run(Graph &graph) {
  bool changed = false;

  for (uint32_t i = 0; i < graph.size(); ++i) {
    auto &n = graph.nodes()[i];
    if (!n.op || n.isRemoved)
      continue;
    if (n.logicalType != LogicalOpType::Reshape)
      continue;
    if (n.inputIds.empty())
      continue;

    uint32_t inputId = n.inputIds[0];
    const auto &inputNode = graph.node(inputId);
    if (n.op->outputShape() == inputNode.op->outputShape()) {
      // This reshape is a no-op — rewire consumers to use the input directly
      graph.replaceAllUses(i, inputId);
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
    if (!n.op || n.isRemoved)
      continue;
    if (n.logicalType != LogicalOpType::Reshape)
      continue;
    if (n.inputIds.empty())
      continue;

    uint32_t inputId = n.inputIds[0];
    auto &inputNode = graph.nodes()[inputId];
    if (!inputNode.op || inputNode.isRemoved)
      continue;
    if (inputNode.logicalType != LogicalOpType::Reshape)
      continue;
    if (inputNode.inputIds.empty())
      continue;

    // Skip the intermediate reshape: point this node at the inner input
    n.inputIds[0] = inputNode.inputIds[0];
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
    if (!n.op || n.isRemoved)
      continue;
    if (n.logicalType != LogicalOpType::Transpose)
      continue;
    if (n.inputIds.empty())
      continue;

    uint32_t inputId = n.inputIds[0];
    const auto &inputNode = graph.nodes()[inputId];
    if (!inputNode.op || inputNode.isRemoved)
      continue;
    if (inputNode.logicalType != LogicalOpType::Transpose)
      continue;
    if (inputNode.inputIds.empty())
      continue;

    // transpose(transpose(x)) → x
    uint32_t origInputId = inputNode.inputIds[0];
    graph.replaceAllUses(i, origInputId);
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
    if (!n.op || n.isRemoved)
      continue;

    // Don't remove outputs or inputs
    if (n.isOutput)
      continue;
    if (n.isInput)
      continue;

    if (n.refCount == 0) {
      // Decrement refCount on this node's inputs
      for (uint32_t inputId : n.inputIds) {
        if (inputId < graph.size()) {
          auto &inputNode = graph.nodes()[inputId];
          if (inputNode.op && inputNode.refCount > 0)
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
