#include "GraphExecutor.h"
#include "OpNode.h"
#include "Operations.h"
#include "Runtime.h"

#include <stdexcept>

namespace cut {
namespace graph {

GraphExecutor::GraphExecutor(Operations &ops, Runtime &runtime)
    : ops_(&ops), runtime_(&runtime) {}

std::vector<Tensor> GraphExecutor::execute(Graph &graph) {
  tensorMap_.clear();
  tensorMap_.resize(graph.size());

  // Execute in topological order
  auto order = graph.topologicalOrder();
  for (uint32_t idx : order) {
    executeNode(graph, idx);
  }

  // Collect outputs
  std::vector<Tensor> results;
  for (const auto &out : graph.outputs()) {
    if (out.isValid()) {
      results.push_back(tensorMap_[out.id]);
    }
  }
  return results;
}

void GraphExecutor::executeNode(Graph &graph, uint32_t nodeIndex) {
  auto &op = graph.nodes()[nodeIndex];
  if (!op || op->isGraphRemoved())
    return;

  // --- Input nodes ---
  if (op->isInputNode()) {
    tensorMap_[nodeIndex] = static_cast<InputOpNode *>(op.get())->gpuHandle();
    return;
  }

  // Resolve real inputs from the tensorMap
  std::vector<Tensor> realInputs;
  for (uint32_t id : op->graphInputIds()) {
    realInputs.push_back(tensorMap_[id]);
  }

  // --- Deferred ops (softmax, variance, layerNorm, etc.) ---
  if (auto *d = dynamic_cast<DeferredOpNode *>(op.get())) {
    tensorMap_[nodeIndex] = d->execute(*ops_, realInputs);
    return;
  }

  // --- Standard OpNode dispatch ---
  op->rebindInputs(realInputs);

  // Reuse the output tensor allocated during graph construction.
  // Creating new tensors each execution can trigger buffer-pool reuse
  // that aliases an output with a still-in-flight input.
  tensorMap_[nodeIndex] = op->output();

  ops_->encodeOp(*op);
}

} // namespace graph
} // namespace cut
