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

  DataType shaderDtype() const override;
  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

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

} // namespace cut
