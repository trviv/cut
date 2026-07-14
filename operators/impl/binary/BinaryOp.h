#pragma once

#include "OpNode.h"

namespace cut {

class BinaryOpNode : public OpNode {
public:
  BinaryOpNode(OperatorEnum op,
               TensorStore &store,
               const Tensor &a,
               const TensorLike &b,
               std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

  /// Returns the scalar bits (only meaningful for VecScalar variant).
  uint32_t scalarBits() const { return scalarBits_; }

private:
  enum class BinaryOpVariant {
    VecVec,         // tensor-tensor (2 buffer bindings)
    VecVecCmp,      // tensor-tensor comparison (typed input, uint output)
    VecScalar,      // tensor-scalar via push constant
    VecScalarCmp,   // tensor-scalar comparison via push constant
    VecScalarBuf,   // tensor-scalar via buffer (second is shape {1})
    VecScalarBufCmp // tensor-scalar buffer comparison
  };

  static bool isComparisonOp(OperatorEnum op);

  DataType dtype_;
  DataType outputDtype_;
  size_t numElements_;
  BinaryOpVariant variant_;
  uint32_t scalarBits_ = 0; // Only used for VecScalar variant
};

/// Row-broadcast binary op: a is [rows, cols] (2-D or higher; the innermost
/// dimension is the broadcast axis), b is a 1-D [cols] vector:
/// out[r, c] = op(a[r, c], b[c]).
class BinaryRowBcastOpNode : public OpNode {
public:
  BinaryRowBcastOpNode(OperatorEnum op,
                       TensorStore &store,
                       const Tensor &a,
                       const Tensor &b,
                       std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  // Distinct tag: shares the binary OperatorEnums with BinaryOpNode but uses
  // a different shader/binding layout (pipeline-cache collision otherwise).
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t numElements_; // rows * alignedCols
  uint32_t cols_;
  uint32_t alignedCols_;
  std::vector<uint32_t> outShape_;
};

} // namespace cut
