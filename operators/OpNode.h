#pragma once

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cut {

class Dispatcher;
class Operations;
class Runtime;

// ============================================================================
// LogicalOpType — coarse classification for graph optimizer passes
// ============================================================================

enum class LogicalOpType { Input, Reshape, Transpose, Other };

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

  /// Returns the input tensor handles.
  const std::vector<Tensor> &inputs() const { return inputs_; }

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

  /// Rebinds input tensor handles for graph execution.
  void rebindInputs(const std::vector<Tensor> &newInputs) {
    inputs_ = newInputs;
  }

  /// Rebinds the output tensor handle for graph execution.
  void rebindOutput(const Tensor &newOutput) { output_ = newOutput; }

  // ==========================================================================
  // Graph metadata — used when this OpNode lives inside a Graph
  // ==========================================================================

  /// Coarse logical type for optimizer passes.
  virtual LogicalOpType logicalType() const { return LogicalOpType::Other; }

  /// Human-readable name for display/reporting.
  virtual std::string displayName() const;

  /// Whether this node is an InputOpNode (graph input, not dispatched to GPU).
  virtual bool isInputNode() const { return false; }

  /// Graph edge indices: IDs of nodes whose outputs feed into this node.
  const std::vector<uint32_t> &graphInputIds() const { return graphInputIds_; }
  void setGraphInputIds(std::vector<uint32_t> ids) {
    graphInputIds_ = std::move(ids);
  }

  /// Reference count: how many other graph nodes consume this node's output.
  uint32_t graphRefCount() const { return graphRefCount_; }
  void setGraphRefCount(uint32_t c) { graphRefCount_ = c; }

  /// Whether this node's output is a graph output.
  bool isGraphOutput() const { return isGraphOutput_; }
  void setGraphOutput(bool v) { isGraphOutput_ = v; }

  /// Tombstone flag for removed/dead nodes.
  bool isGraphRemoved() const { return isGraphRemoved_; }
  void setGraphRemoved(bool v) { isGraphRemoved_ = v; }

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
  std::vector<Tensor> inputs_{};
  Tensor output_{};
  std::vector<std::unique_ptr<OpNode>> subOps_{};

  // Graph metadata
  std::vector<uint32_t> graphInputIds_;
  uint32_t graphRefCount_ = 0;
  bool isGraphOutput_ = false;
  bool isGraphRemoved_ = false;
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

// ============================================================================
// InputOpNode — represents a graph input (pre-existing GPU tensor)
// ============================================================================

class InputOpNode : public OpNode {
public:
  InputOpNode(const Tensor &gpuHandle,
              const std::vector<uint32_t> &shape,
              DataType dtype,
              bool isConstant = false);

  bool isInputNode() const override { return true; }
  LogicalOpType logicalType() const override { return LogicalOpType::Input; }
  std::string displayName() const override;

  DataType shaderDtype() const override { return dtype_; }
  std::vector<uint32_t> outputShape() const override { return shape_; }
  DataType outputDtype() const override { return dtype_; }
  ThreadSize dispatchSize() const override { return {0, 0, 0}; }

  const Tensor &gpuHandle() const { return gpuHandle_; }
  void setGpuHandle(const Tensor &h) { gpuHandle_ = h; }
  bool isConstant() const { return isConstant_; }

protected:
  std::vector<uint8_t> pushConstants() const override { return {}; }

private:
  Tensor gpuHandle_;
  std::vector<uint32_t> shape_;
  DataType dtype_;
  bool isConstant_;
};

// ============================================================================
// DeferredOpNode — defers execution to Operations at graph execution time
// ============================================================================

/// For operations (variance, softmax, etc.) that compute on CPU and can't be
/// represented as a single GPU dispatch. Stores a callable that re-dispatches
/// through the Operations API at execution time.
class DeferredOpNode : public OpNode {
public:
  using ExecuteFn =
      std::function<Tensor(Operations &, const std::vector<Tensor> &)>;

  DeferredOpNode(const std::vector<uint32_t> &shape,
                 DataType dtype,
                 std::string name,
                 ExecuteFn fn);

  std::string displayName() const override { return name_; }

  DataType shaderDtype() const override { return dtype_; }
  std::vector<uint32_t> outputShape() const override { return shape_; }
  DataType outputDtype() const override { return dtype_; }
  ThreadSize dispatchSize() const override { return {0, 0, 0}; }

  Tensor execute(Operations &ops, const std::vector<Tensor> &inputs);

protected:
  std::vector<uint8_t> pushConstants() const override { return {}; }

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  std::string name_;
  ExecuteFn fn_;
};

// ============================================================================
// StubOpNode — non-executable placeholder for cloned/reporting graphs
// ============================================================================

class StubOpNode : public OpNode {
public:
  StubOpNode(OperatorEnum opEnum,
             const std::vector<uint32_t> &shape,
             DataType dtype,
             std::string name,
             std::string detail,
             bool isConstant = false);

  bool isInputNode() const override { return isInput_; }
  LogicalOpType logicalType() const override { return logicalType_; }
  std::string displayName() const override { return name_; }

  DataType shaderDtype() const override { return dtype_; }
  std::vector<uint32_t> outputShape() const override { return shape_; }
  DataType outputDtype() const override { return dtype_; }
  ThreadSize dispatchSize() const override { return {0, 0, 0}; }

  const std::string &detail() const { return detail_; }
  bool isConstant() const { return isConstant_; }

  void setIsInput(bool v) { isInput_ = v; }
  void setLogicalType(LogicalOpType t) { logicalType_ = t; }

protected:
  std::vector<uint8_t> pushConstants() const override { return {}; }

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  std::string name_;
  std::string detail_;
  bool isConstant_ = false;
  bool isInput_ = false;
  LogicalOpType logicalType_ = LogicalOpType::Other;
};

} // namespace cut
