#pragma once

#include <cstdint>

namespace cut {
namespace graph {

// ============================================================================
// VirtualTensor — lightweight proxy for a tensor that doesn't exist on GPU yet
// ============================================================================

/// Index into Graph::nodes_. Represents the output of one graph node.
struct VirtualTensor {
  uint32_t id = UINT32_MAX;

  bool isValid() const { return id != UINT32_MAX; }
  bool operator==(const VirtualTensor &o) const { return id == o.id; }
  bool operator!=(const VirtualTensor &o) const { return id != o.id; }
};

} // namespace graph
} // namespace cut
