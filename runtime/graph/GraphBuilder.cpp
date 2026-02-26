#include "GraphBuilder.h"
#include "Operations.h"
#include "Runtime.h"

namespace cut {
namespace graph {

GraphBuilder::GraphBuilder(Runtime &runtime) : ops_(&runtime.ops()) {
  ops_->setGraph(&graph_);
}

GraphBuilder::~GraphBuilder() {
  // Ensure we exit graph mode if build() was never called
  if (ops_->isGraphMode()) {
    ops_->clearGraph();
  }
}

Graph GraphBuilder::build() {
  ops_->clearGraph();
  return std::move(graph_);
}

Tensor GraphBuilder::input(const Tensor &gpuHandle, bool isConstant) {
  return ops_->registerInput(gpuHandle, isConstant);
}

void GraphBuilder::markOutput(const Tensor &t) {
  graph_.markOutput(t);
}

} // namespace graph
} // namespace cut
