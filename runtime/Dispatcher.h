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

/**
 * Dispatcher class for encoding math operators to the compute backend.
 * Generates compute dispatches based on operator enums, inferring dtype
 * and workgroup size from buffer bindings.
 *
 * Also manages temporary GPU buffers for multi-pass operations
 * (multi-workgroup reduce, prefix scan, sort).
 */
class Dispatcher {
public:
  /**
   * Constructs a Dispatcher with the given compute interface.
   * @param iface The compute interface to use for encoding dispatches.
   */
  explicit Dispatcher(ComputeInterface *iface);

  /**
   * Encodes an operator dispatch using an OpNode.
   * The node provides its own shader SPIR-V via shader() and cache key
   * via shaderKey(), so the Dispatcher doesn't need to branch on node type.
   *
   * @param node The operator node with all dispatch information.
   * @return true if commands were encoded, false if skipped (e.g. sort no-op).
   */
  bool encode(std::unique_ptr<OpNode> node);

  /**
   * Releases all temporary buffers back to the pool.
   * Should be called after a multi-pass operation completes.
   */
  void releaseTempBuffers();

  /**
   * Acquires a temporary GPU buffer from the pool or creates a new one.
   * Used by multi-pass OpNodes to allocate intermediate buffers.
   * @param numElements Number of elements.
   * @param dtype Data type of elements.
   * @return Handle to the temporary buffer.
   */
  Tensor acquireTempBuffer(size_t numElements, DataType dtype);

private:
  ComputeInterface *iface_;

  /// Pool of reusable temporary GPU buffers.
  std::vector<Tensor> tempBufferPool_;

  /// Temporary buffers currently in use by the current multi-pass operation.
  std::vector<Tensor> activeTempBuffers_;

  /// Shader cache keyed by OpNode::shaderKey().
  std::unordered_map<size_t, Tensor> shaderCache_;

  /**
   * Encodes a compute-to-compute barrier.
   * Ensures prior shader writes are visible to subsequent shader reads.
   */
  void encodeBarrier();

  /**
   * Gets or creates a cached shader module for an OpNode.
   * Uses node.shaderKey() for caching and node.shader() on cache miss.
   */
  Tensor getOrCreateShader(const OpNode &node);
};

} // namespace cut
