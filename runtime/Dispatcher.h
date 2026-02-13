#pragma once

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace cut {

// Forward declarations
class ComputeInterface;

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
   * Encodes a math operator dispatch to the compute backend.
   *
   * @param op The operator to execute (from OperatorEnum).
   * @param bindings Vector of compute bindings (buffers and data).
   * @param shader Pre-created shader handle for this operator.
   * @param executionSize Number of elements to process (from buffer execution
   * size).
   */
  void encode(OperatorEnum op,
              const std::vector<ComputeBinding> &bindings,
              const ComputeHandle &shader,
              size_t executionSize,
              DataType dtype = DataType::Float32);

  /**
   * Releases all temporary buffers back to the pool.
   * Should be called after a multi-pass operation completes.
   */
  void releaseTempBuffers();

private:
  ComputeInterface *iface_;

  /// Pool of reusable temporary GPU buffers.
  std::vector<ComputeHandle> tempBufferPool_;

  /// Temporary buffers currently in use by the current multi-pass operation.
  std::vector<ComputeHandle> activeTempBuffers_;

  /// Internal shader cache keyed by hash of GLSL source.
  std::unordered_map<size_t, ComputeHandle> internalShaderCache_;

  /**
   * Acquires a temporary GPU buffer from the pool or creates a new one.
   * @param numElements Number of elements.
   * @param dtype Data type of elements.
   * @return Handle to the temporary buffer.
   */
  ComputeHandle acquireTempBuffer(size_t numElements, DataType dtype);

  /**
   * Encodes a compute-to-compute barrier.
   * Ensures prior shader writes are visible to subsequent shader reads.
   */
  void encodeBarrier();

  /**
   * Gets or creates an internal shader from GLSL source.
   * Compiles the source to SPIR-V and caches the result.
   * @param glslSource The GLSL source code.
   * @return Handle to the shader module.
   */
  ComputeHandle getOrCreateInternalShader(const std::string &glslSource);

  /**
   * Gets or creates an internal shader by OperatorEnum.
   * Uses the shader generation system (getShader) to compile, then caches.
   * Only works for non-parameterized internal shader templates.
   * @param op The internal operator enum (e.g., InternalScanPerWg).
   * @return Handle to the shader module.
   */
  ComputeHandle getOrCreateInternalShader(OperatorEnum op);

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
  void dispatchInternal(const ComputeHandle &shader,
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
