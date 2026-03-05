#pragma once

#include "OpNode.h"

namespace cut {

enum class FusedBinaryVariant {
  VecScalarVecVec,    // op2(op1(A, scalar), B)
  VecVecVecScalar,    // op2(op1(A, B), scalar)
  VecVecVecScalarBuf, // op2(op1(A, B), scalarBuf[0])
  VecScalarBufVecVec, // op2(op1(A, scalarBuf[0]), B)
  UnaryVecVec,        // binaryOp(unary(A), B)
  VecVecUnary         // unary(binaryOp(A, B))
};

class FusedBinaryOpNode : public OpNode {
public:
  // For VecScalarVecVec and VecVecVecScalar (scalar in push constants)
  FusedBinaryOpNode(TensorStore &store,
                    OperatorEnum op1,
                    OperatorEnum op2,
                    const Tensor &a,
                    const Tensor &b,
                    uint32_t scalarBits,
                    FusedBinaryVariant variant);

  // For UnaryVecVec and VecVecUnary (no scalar, 2 buffer inputs)
  FusedBinaryOpNode(TensorStore &store,
                    OperatorEnum op1,
                    OperatorEnum op2,
                    const Tensor &a,
                    const Tensor &b,
                    FusedBinaryVariant variant);

  // For VecVecVecScalarBuf and VecScalarBufVecVec (3 buffer inputs)
  FusedBinaryOpNode(TensorStore &store,
                    OperatorEnum op1,
                    OperatorEnum op2,
                    const Tensor &a,
                    const Tensor &b,
                    const Tensor &scalarBuf,
                    FusedBinaryVariant variant);

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::string displayName() const override;

private:
  OperatorEnum op1_;
  OperatorEnum op2_;
  DataType dtype_;
  size_t numElements_;
  uint32_t scalarBits_;
  FusedBinaryVariant variant_;
};

} // namespace cut
