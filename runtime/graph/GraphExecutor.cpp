#include "GraphExecutor.h"
#include "MemoryPlanner.h"
#include "OpNode.h"
#include "Operations.h"
#include "TensorStore.h"
#include "impl/memory/MemoryOp.h"

#include <stdexcept>

namespace cut {
namespace graph {

GraphExecutor::GraphExecutor(Operations &ops, TensorStore &store)
    : ops_(&ops), store_(&store) {}

std::vector<Tensor> GraphExecutor::execute(Graph &graph) {
  // Plan transient tensor memory before execution
  MemoryPlanner planner(*store_);
  planner.plan(graph);

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
  auto &gn = graph.nodes()[nodeIndex];
  if (!gn.op || gn.isRemoved)
    return;

  // --- Input nodes ---
  if (gn.isInput) {
    tensorMap_[nodeIndex] =
        static_cast<InputOpNode *>(gn.op.get())->gpuHandle();
    return;
  }

  // Resolve real inputs from the tensorMap
  std::vector<Tensor> realInputs;
  for (uint32_t id : gn.inputIds) {
    realInputs.push_back(tensorMap_[id]);
  }

  // --- Zero-copy slice: create a view into parent, no GPU dispatch ---
  auto *sliceOp = dynamic_cast<SliceOpNode *>(gn.op.get());
  if (sliceOp) {
    // Parent is the single input — create a buffer view at the slice offset.
    Tensor parent = realInputs[0];
    auto shape = sliceOp->outputShape();
    auto dtype = sliceOp->outputDtype();
    size_t offset = sliceOp->byteOffset();
    Tensor view = store_->createTensorView(parent, offset, shape, dtype);
    gn.op->rebindOutput(view);
    tensorMap_[nodeIndex] = view;
    // The view shares the parent's VkBuffer but has a different ComputeHandle,
    // so the barrier tracker won't see it as the same buffer. Insert a barrier
    // to ensure prior writes to the parent are visible to subsequent reads
    // from the view.
    ops_->barrier();
    return;
  }

  // --- Standard OpNode dispatch ---
  gn.op->rebindInputs(realInputs);

  // Reuse the output tensor allocated during graph construction.
  // Creating new tensors each execution can trigger buffer-pool reuse
  // that aliases an output with a still-in-flight input.
  tensorMap_[nodeIndex] = gn.op->output();

  ops_->dispatch(*gn.op);
}

} // namespace graph
} // namespace cut
