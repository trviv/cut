#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cut {

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

  /// Returns the DataType for shader dtype selection.
  virtual DataType shaderDtype() const = 0;

  /// Returns the variant index (-1 = not applicable).
  virtual int variantIndex() const { return -1; }

  /// Returns the computed output shape.
  virtual std::vector<uint32_t> outputShape() const = 0;

  /// Returns the output dtype (defaults to shaderDtype()).
  virtual DataType outputDtype() const { return shaderDtype(); }

  /// Returns the dispatch thread dimensions (total grid size).
  virtual ThreadSize dispatchSize() const = 0;

  /// Returns push constant data as a byte vector.
  virtual std::vector<uint8_t> pushConstants() const = 0;

  /// Returns true if this op requires multi-pass dispatch (scan, sort, etc.).
  virtual bool isMultiPass() const { return false; }

  /// Returns true if this op needs a dim-reduce shader (looked up internally
  /// by Dispatcher).
  virtual bool isDimReduce() const { return false; }

  /// Returns the base reduce op for dim-reduce ops (used by Dispatcher to
  /// look up the right dim-reduce shader variant).
  virtual OperatorEnum baseReduceOp() const { return op(); }

  /// Returns the execution size (for multi-WG reduce threshold, etc.).
  virtual size_t executionSize() const;

  /// Sets buffer handles for input and output bindings.
  /// Called by Operations after creating the output tensor.
  void setHandles(std::vector<Tensor> inputs, Tensor output);

  /// Overload for in-place ops (sort) with no separate output.
  void setHandles(std::vector<Tensor> inputs);

  /// Returns the ComputeBinding vector for dispatch encoding.
  virtual std::vector<ComputeBinding> handleBindings() const;

protected:
  explicit OpNode(OperatorEnum op) : op_(op) {}

  OperatorEnum op_;
  std::vector<Tensor> inputs_;
  Tensor output_;
  bool hasOutput_ = false;
};

} // namespace cut
