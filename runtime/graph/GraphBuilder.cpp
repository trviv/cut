#include "GraphBuilder.h"
#include "Operations.h"
#include "Runtime.h"

namespace cut {
namespace graph {

GraphBuilder::GraphBuilder(Runtime &runtime) : ops_(&runtime.ops()) {
  ops_->flush();
}

GraphBuilder::~GraphBuilder() {
  // Discard any recorded ops if build() was never called
  if (!built_) {
    ops_->takeGraph();
  }
}

std::unique_ptr<Graph> GraphBuilder::build() {
  built_ = true;
  return ops_->takeGraph();
}

Tensor GraphBuilder::input(const Tensor &gpuHandle, bool isConstant) {
  return ops_->registerInput(gpuHandle, isConstant);
}

void GraphBuilder::markOutput(const Tensor &t) {
  ops_->markGraphOutput(t);
}

} // namespace graph
} // namespace cut
