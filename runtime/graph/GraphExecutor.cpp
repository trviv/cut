#include "GraphExecutor.h"
#include "OpNode.h"
#include "Operations.h"

#include <stdexcept>

namespace cut {
namespace graph {

GraphExecutor::GraphExecutor(Operations &ops) : ops_(&ops) {}

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
  for (uint32_t out : graph.outputs()) {
    if (out != UINT32_MAX) {
      results.push_back(tensorMap_[out]);
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

  // --- Standard OpNode dispatch ---
  op->rebindInputs(realInputs);

  // Reuse the output tensor allocated during graph construction.
  // Creating new tensors each execution can trigger buffer-pool reuse
  // that aliases an output with a still-in-flight input.
  tensorMap_[nodeIndex] = op->output();

  ops_->dispatch(*op);
}

} // namespace graph
} // namespace cut
