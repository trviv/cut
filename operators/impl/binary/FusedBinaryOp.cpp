#include "FusedBinaryOp.h"
#include "BinaryShaders.generated.h"
#include "Shaders.h"
#include "TensorStore.h"

namespace cut {

// Constructor for VecScalarVecVec and VecVecVecScalar (scalar in push
// constants)
FusedBinaryOpNode::FusedBinaryOpNode(TensorStore &store,
                                     OperatorEnum op1,
                                     OperatorEnum op2,
                                     const Tensor &a,
                                     const Tensor &b,
                                     uint32_t scalarBits,
                                     FusedBinaryVariant variant)
    : OpNode(FusedBinary, store), op1_(op1), op2_(op2), scalarBits_(scalarBits),
      variant_(variant) {
  const auto &bufA = store.getTensor(a);
  dtype_ = bufA.getDtype();
  numElements_ = alignedElementCount(bufA.getShape());

  inputs_ = {a, b};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

// Constructor for UnaryVecVec and VecVecUnary (no scalar, 2 buffer inputs)
FusedBinaryOpNode::FusedBinaryOpNode(TensorStore &store,
                                     OperatorEnum op1,
                                     OperatorEnum op2,
                                     const Tensor &a,
                                     const Tensor &b,
                                     FusedBinaryVariant variant)
    : OpNode(FusedBinary, store), op1_(op1), op2_(op2), scalarBits_(0),
      variant_(variant) {
  const auto &bufA = store.getTensor(a);
  dtype_ = bufA.getDtype();
  numElements_ = alignedElementCount(bufA.getShape());

  inputs_ = {a, b};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

// Constructor for VecVecVecScalarBuf and VecScalarBufVecVec (3 buffer inputs)
FusedBinaryOpNode::FusedBinaryOpNode(TensorStore &store,
                                     OperatorEnum op1,
                                     OperatorEnum op2,
                                     const Tensor &a,
                                     const Tensor &b,
                                     const Tensor &scalarBuf,
                                     FusedBinaryVariant variant)
    : OpNode(FusedBinary, store), op1_(op1), op2_(op2), scalarBits_(0),
      variant_(variant) {
  const auto &bufA = store.getTensor(a);
  dtype_ = bufA.getDtype();
  numElements_ = alignedElementCount(bufA.getShape());

  if (variant == FusedBinaryVariant::VecScalarBufVecVec) {
    inputs_ = {a, scalarBuf, b}; // binding: A, scalarBuf, B, out
  } else {
    inputs_ = {a, b, scalarBuf}; // binding: A, B, scalarBuf, out
  }
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType FusedBinaryOpNode::outputDtype() const {
  return dtype_;
}

size_t FusedBinaryOpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(dtype_) & 0xF) << 16;
  key |= (static_cast<size_t>(op1_) & 0xFF) << 20;
  key |= (static_cast<size_t>(op2_) & 0xFF) << 28;
  key |= static_cast<size_t>(variant_) << 36;
  return key;
}

std::optional<std::vector<uint32_t>> FusedBinaryOpNode::shader() const {
  std::optional<std::vector<uint32_t>> compiled;

  switch (variant_) {
  case FusedBinaryVariant::VecScalarVecVec:
    compiled = compiledBinaryFusedVecScalarVecVec(dtype_, dtype_, dtype_);
    break;
  case FusedBinaryVariant::VecVecVecScalar:
    compiled = compiledBinaryFusedVecVecVecScalar(dtype_, dtype_, dtype_);
    break;
  case FusedBinaryVariant::VecVecVecScalarBuf:
    compiled = compiledBinaryFusedVecVecVecScalarBuf(dtype_, dtype_, dtype_);
    break;
  case FusedBinaryVariant::VecScalarBufVecVec:
    compiled = compiledBinaryFusedVecScalarBufVecVec(dtype_, dtype_, dtype_);
    break;
  case FusedBinaryVariant::UnaryVecVec:
    compiled = compiledBinaryFusedUnaryVecVec(dtype_, dtype_, dtype_);
    break;
  case FusedBinaryVariant::VecVecUnary:
    compiled = compiledBinaryFusedVecVecUnary(dtype_, dtype_, dtype_);
    break;
  }

  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstants(spirv, {{1, static_cast<uint32_t>(op1_)},
                               {2, static_cast<uint32_t>(op2_)}});
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> FusedBinaryOpNode::outputShape() const {
  return store_->getTensor(inputs_[0]).getShape();
}

ThreadSize FusedBinaryOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> FusedBinaryOpNode::pushConstants() const {
  switch (variant_) {
  case FusedBinaryVariant::VecScalarVecVec:
  case FusedBinaryVariant::VecVecVecScalar: {
    struct PushConstants {
      uint32_t numElements;
      uint32_t scalarBits;
    } pc{static_cast<uint32_t>(numElements_), scalarBits_};
    return toBytes(pc);
  }
  default: {
    uint32_t n = static_cast<uint32_t>(numElements_);
    return toBytes(n);
  }
  }
}

std::string FusedBinaryOpNode::displayName() const {
  return "FusedBinary";
}

} // namespace cut
