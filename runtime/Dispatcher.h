#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace cut {

// Forward declarations
class ComputeInterface;
class OpNode;
class TensorStore;

/**
 * Dispatcher class for encoding math operators to the compute backend.
 * Generates compute dispatches based on operator enums, inferring dtype
 * and workgroup size from buffer bindings.
 */
class Dispatcher {
public:
  /**
   * Constructs a Dispatcher with the given tensor store.
   * @param store The tensor store (provides iface() for encoding and temp
   * buffers).
   */
  explicit Dispatcher(TensorStore *store);

  /**
   * Encodes an operator dispatch using an OpNode.
   * The node provides its own shader SPIR-V via shader() and cache key
   * via shaderKey(), so the Dispatcher doesn't need to branch on node type.
   *
   * @param node The operator node with all dispatch information.
   * @return true if commands were encoded, false if skipped (e.g. sort no-op).
   */
  bool encode(std::unique_ptr<OpNode> node);

  /// Non-owning overload for graph execution (graph retains node ownership).
  bool encode(OpNode &node);

  /// Encodes a compute-to-compute barrier.
  /// Ensures prior shader writes are visible to subsequent shader reads.
  void encodeBarrier();

private:
  TensorStore *store_;
  ComputeInterface *iface_;

  /// Shader cache keyed by OpNode::shaderKey().
  std::unordered_map<size_t, Tensor> shaderCache_;

  /**
   * Gets or creates a cached shader module for an OpNode.
   * Uses node.shaderKey() for caching and node.shader() on cache miss.
   */
  Tensor getOrCreateShader(const OpNode &node);
};

} // namespace cut
