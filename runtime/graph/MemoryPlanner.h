#pragma once

#include "Graph.h"

#include <cstddef>

namespace cut {

class TensorStore;

namespace graph {

/// Performs liveness-based memory planning for transient graph tensors.
/// After graph optimization, replaces eagerly-allocated transient outputs
/// with views into a single shared arena buffer, minimizing peak memory.
class MemoryPlanner {
public:
  explicit MemoryPlanner(TensorStore &store);

  /// Analyze the graph and replace transient tensor outputs with arena views.
  /// Must be called after optimization and before execution.
  /// Returns the number of bytes saved (original total - arena size),
  /// or 0 if no savings were achieved.
  size_t plan(Graph &graph);

private:
  TensorStore *store_;
};

} // namespace graph
} // namespace cut
