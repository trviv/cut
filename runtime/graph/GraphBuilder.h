#pragma once

#include "Graph.h"

#include <ComputeHandle.h>
#include <memory>

namespace cut {

class Operations;
class Runtime;

namespace graph {

/// Builds a computation graph by routing Operations calls through graph mode.
/// Instead of dispatching to the GPU, each call to ops() creates an OpNode via
/// Operations and returns the resulting Tensor. After construction, call
/// build() to obtain the Graph for optimization and execution.
class GraphBuilder {
public:
  explicit GraphBuilder(Runtime &runtime);
  ~GraphBuilder();

  /// Move-returns the constructed graph. The builder should not be used after
  /// this.
  std::unique_ptr<Graph> build();

  /// Returns the Operations instance in graph mode.
  /// Use this to call any operation (e.g. builder.ops().matmul(a, b)).
  Operations &ops() { return *ops_; }

  /// Register a pre-existing GPU tensor as a graph input.
  /// Set isConstant=true for model weights (enables optimizer reasoning).
  Tensor input(const Tensor &gpuHandle, bool isConstant = false);

  /// Mark a tensor as a graph output.
  void markOutput(const Tensor &t);

private:
  Operations *ops_;
  bool built_ = false;
};

} // namespace graph
} // namespace cut
