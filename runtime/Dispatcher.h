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
 * Also manages temporary GPU buffers and internal shaders for multi-pass
 * operations (multi-workgroup reduce, prefix scan, sort).
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
   * Handles shader resolution, sort early-out, multi-pass, and dim-reduce
   * internally.
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

private:
  ComputeInterface *iface_;

  /// Pool of reusable temporary GPU buffers.
  std::vector<Tensor> tempBufferPool_;

  /// Temporary buffers currently in use by the current multi-pass operation.
  std::vector<Tensor> activeTempBuffers_;

  /// Internal shader cache keyed by (op, dtype) composite key.
  std::unordered_map<size_t, Tensor> internalShaderCache_;

  /**
   * Acquires a temporary GPU buffer from the pool or creates a new one.
   * @param numElements Number of elements.
   * @param dtype Data type of elements.
   * @return Handle to the temporary buffer.
   */
  Tensor acquireTempBuffer(size_t numElements, DataType dtype);

  /**
   * Encodes a compute-to-compute barrier.
   * Ensures prior shader writes are visible to subsequent shader reads.
   */
  void encodeBarrier();

  /**
   * Gets or creates an internal shader by OperatorEnum.
   * Uses the shader generation system (getShader) to compile, then caches.
   * @param op The operator enum (e.g., InternalScanPerWg).
   * @param dtype Data type for dtype-parameterized shaders.
   * @return Handle to the shader module.
   */
  Tensor getOrCreateInternalShader(OperatorEnum op,
                                   DataType dtype = DataType::Float32);

  /**
   * Gets or creates a cached shader for a standard operator.
   * If spec is provided, looks up/creates a spec-specific shader.
   */
  Tensor getOrCreateShader(OperatorEnum op,
                           DataType dtype,
                           std::optional<uint32_t> spec);

  /**
   * Gets or creates a dim-wise reduction shader for a base reduce op.
   * Uses getDimReduceShader() to compile the ReduceDim/ReduceDimArg template.
   */
  Tensor getOrCreateDimReduceShader(OperatorEnum reduceOp,
                                    DataType dtype = DataType::Float32,
                                    std::optional<uint32_t> spec = {});

  /**
   * Dispatches an internal shader with the given bindings and push constants.
   * Handles shader lookup, dispatch creation, push constant binding, and
   * encoding in a single call to reduce boilerplate in multi-pass operations.
   *
   * @param shader Pre-resolved shader handle.
   * @param bindings Buffer bindings for the dispatch.
   * @param threadSize Workgroup thread dimensions.
   * @param pushData Push constant data bound after all buffer bindings.
   */
  void dispatchInternal(const Tensor &shader,
                        const std::vector<ComputeBinding> &bindings,
                        ThreadSize threadSize,
                        const DataReference &pushData);

  /**
   * Dispatches an internal shader by OperatorEnum.
   * Resolves the shader via getOrCreateInternalShader, then dispatches.
   */
  void dispatchInternal(OperatorEnum op,
                        const std::vector<ComputeBinding> &bindings,
                        ThreadSize threadSize,
                        const DataReference &pushData);

  /**
   * Encodes a multi-workgroup reduction operation.
   * Two-phase: partial reduce across workgroups, barrier, final reduce.
   */
  void encodeMultiWorkgroupReduce(OperatorEnum op,
                                  const std::vector<ComputeBinding> &bindings,
                                  size_t executionSize);

  /**
   * Encodes a prefix scan operation (exclusive or inclusive sum).
   * Two-phase for large inputs: per-workgroup scan, then propagate prefixes.
   */
  void encodePrefixScan(OperatorEnum op,
                        const std::vector<ComputeBinding> &bindings,
                        size_t executionSize);

  /**
   * Encodes a bitonic sort operation (multi-pass compare-and-swap).
   */
  void encodeBitonicSort(const std::vector<ComputeBinding> &bindings,
                         size_t executionSize);

  /**
   * Encodes a radix sort operation (multi-pass histogram + scatter).
   */
  void encodeRadixSort(const std::vector<ComputeBinding> &bindings,
                       size_t executionSize);
};

} // namespace cut
