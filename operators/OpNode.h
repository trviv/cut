#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace cut {

class Dispatcher;
class Runtime;

// ============================================================================
// Utility functions shared across OpNode subclasses
// ============================================================================

/// Computes aligned element count (rounds innermost dim to multiple of 4).
size_t alignedElementCount(const std::vector<uint32_t> &shape);

/// Computes actual (unpadded) element count from a shape vector.
size_t actualElementCount(const std::vector<uint32_t> &shape);

/// Converts a POD struct/value to a byte vector for push constants.
template <typename T>
inline std::vector<uint8_t> toBytes(const T &val) {
  std::vector<uint8_t> bytes(sizeof(T));
  std::memcpy(bytes.data(), &val, sizeof(T));
  return bytes;
}

/// Appends bytes of a POD value to an existing byte vector.
template <typename T>
inline void appendBytes(std::vector<uint8_t> &vec, const T &val) {
  size_t offset = vec.size();
  vec.resize(offset + sizeof(T));
  std::memcpy(vec.data() + offset, &val, sizeof(T));
}

// ============================================================================
// OpNode base class
// ============================================================================

/**
 * Base class for all operator nodes.
 * Encapsulates output shape computation, dispatch sizing,
 * and push constant generation for a single GPU operator invocation.
 *
 * Subclasses validate inputs in their constructors, then are passed
 * to Runtime::encodeOperator() for dispatch.
 */
class OpNode {
public:
  virtual ~OpNode() = default;

  /// Returns the OperatorEnum for shader lookup.
  OperatorEnum op() const { return op_; }

  /// Returns the output tensor handle.
  const Tensor &output() const { return output_; }

  /// Returns the DataType for shader dtype selection.
  virtual DataType shaderDtype() const = 0;

  /// Returns the optional specialization index.
  virtual std::optional<uint32_t> spec() const { return spec_; }

  /// Returns SPIR-V bytecode for this node's shader.
  /// Default calls getShader(op_, shaderDtype()). Override for variant ops
  /// (MatMul, Transpose, etc.) and dim-reduce ops.
  virtual std::optional<std::vector<uint32_t>> shader() const;

  /// Returns a cache key for the Dispatcher's shader cache.
  /// Default: op | (dtype << 16) | (spec << 32).
  virtual size_t shaderKey() const;

  /// Returns the computed output shape.
  virtual std::vector<uint32_t> outputShape() const = 0;

  /// Returns the output dtype (defaults to shaderDtype()).
  virtual DataType outputDtype() const { return shaderDtype(); }

  /// Returns the dispatch thread dimensions (total grid size).
  virtual ThreadSize dispatchSize() const = 0;

  /// Returns true if this op requires multi-pass dispatch (scan, sort, etc.).
  virtual bool isMultiPass() const { return false; }

  /// Returns the execution size (for multi-WG reduce threshold, etc.).
  virtual size_t executionSize() const;

  /// Returns the ComputeBinding vector for dispatch encoding.
  /// Includes tensor bindings (inputs + output) followed by push constants.
  virtual std::vector<ComputeBinding> bindings() const;

  /// Populates and returns the sub-operations for composite/multi-pass ops.
  /// On first call, invokes buildSubOperations() to populate subOps_.
  /// Subsequent calls return the cached result.
  const std::vector<std::unique_ptr<OpNode>> &
  subOperations(Dispatcher &dispatcher);

  /// Whether a barrier should be encoded after dispatching this node.
  virtual bool needsBarrierAfter() const { return false; }

protected:
  OpNode(OperatorEnum op, Runtime &runtime, std::optional<uint32_t> spec = {})
      : op_(op), runtime_(&runtime), spec_(spec) {}

  /// Minimal constructor for internal nodes that don't need Runtime access.
  explicit OpNode(OperatorEnum op) : op_(op), runtime_(nullptr) {}

  /// Returns push constant data as a byte vector.
  virtual std::vector<uint8_t> pushConstants() const = 0;

  /// Override in multi-pass subclasses to populate subOps_.
  virtual void buildSubOperations(Dispatcher &dispatcher) {}

  OperatorEnum op_;
  Runtime *runtime_;
  std::optional<uint32_t> spec_;
  std::vector<Tensor> inputs_;
  Tensor output_;
  bool hasOutput_ = false;
  std::vector<std::unique_ptr<OpNode>> subOps_;
};

// ============================================================================
// InternalOpNode — concrete OpNode for intermediate dispatches
// ============================================================================

/**
 * Represents a single internal GPU dispatch within a multi-pass operation.
 * Does not create output tensors or access Runtime — simply describes
 * the shader, bindings, thread size, push constants, and barrier requirement.
 */
class InternalOpNode : public OpNode {
public:
  InternalOpNode(OperatorEnum op,
                 DataType dtype,
                 std::vector<Tensor> inputs,
                 ThreadSize threadSize,
                 std::vector<uint8_t> pushConstants,
                 bool barrierAfter = false);

  DataType shaderDtype() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  bool needsBarrierAfter() const override;

private:
  DataType dtype_;
  ThreadSize threadSize_;
  std::vector<uint8_t> pushConstants_;
  bool barrierAfter_;
};

} // namespace cut
