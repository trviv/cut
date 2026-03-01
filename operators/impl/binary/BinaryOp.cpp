#include "BinaryOp.h"
#include "BinaryShaders.generated.h"
#include "Shaders.h"
#include "TensorStore.h"

namespace cut {

bool BinaryOpNode::isComparisonOp(OperatorEnum op) {
  return op >= BinaryEqual && op <= BinaryGreaterEqual;
}

BinaryOpNode::BinaryOpNode(OperatorEnum op,
                           TensorStore &store,
                           const Tensor &a,
                           const TensorLike &b,
                           std::optional<uint32_t> spec)
    : OpNode(op, store, spec) {
  const auto &bufA = store.getTensor(a);
  const auto shapeA = bufA.getShape();
  dtype_ = bufA.getDtype();
  numElements_ = alignedElementCount(shapeA);

  bool isCmp = isComparisonOp(op);

  // Detect variant based on b's type
  if (b.isHandle()) {
    const auto &bufB = store.getTensor(b.getHandle());
    const auto shapeB = bufB.getShape();

    if (actualElementCount(shapeB) == 1) {
      variant_ = isCmp ? BinaryOpVariant::VecScalarBufCmp
                       : BinaryOpVariant::VecScalarBuf;
      inputs_ = {a, b.getHandle()};
    } else {
      variant_ = isCmp ? BinaryOpVariant::VecVecCmp : BinaryOpVariant::VecVec;
      inputs_ = {a, b.getHandle()};

      // Validate shapes match
      if (actualElementCount(shapeA) != actualElementCount(shapeB)) {
        throw std::runtime_error(
            "Size mismatch: " + std::to_string(actualElementCount(shapeA)) +
            " vs " + std::to_string(actualElementCount(shapeB)));
      }
    }
  } else if (b.isScalar()) {
    variant_ =
        isCmp ? BinaryOpVariant::VecScalarCmp : BinaryOpVariant::VecScalar;
    scalarBits_ = b.getScalar<uint32_t>();
    inputs_ = {a};
  } else if (b.isData()) {
    variant_ =
        isCmp ? BinaryOpVariant::VecScalarCmp : BinaryOpVariant::VecScalar;
    const auto &data = b.getData();
    // Extract scalar bits based on dtype
    if (dtype_ == DataType::Float32) {
      float val;
      std::memcpy(&val, data.data(), sizeof(float));
      std::memcpy(&scalarBits_, &val, sizeof(float));
    } else {
      std::memcpy(&scalarBits_, data.data(),
                  std::min(data.size(), sizeof(uint32_t)));
    }
    inputs_ = {a};
  } else {
    throw std::runtime_error("Invalid TensorLike type for binary operation");
  }

  // Cmp variants produce UInt32 output; others keep input dtype
  outputDtype_ = isCmp ? DataType::UInt32 : dtype_;

  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType BinaryOpNode::shaderDtype() const {
  return dtype_;
}

DataType BinaryOpNode::outputDtype() const {
  return outputDtype_;
}

size_t BinaryOpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(dtype_) & 0xF) << 16;
  key |= (static_cast<size_t>(outputDtype_) & 0xF) << 20;
  key |= static_cast<size_t>(spec_.value_or(0)) << 48;
  key |= static_cast<size_t>(variant_) << 60;
  return key;
}

std::optional<std::vector<uint32_t>> BinaryOpNode::shader() const {
  std::optional<std::vector<uint32_t>> compiled;

  switch (variant_) {
  case BinaryOpVariant::VecVec:
    compiled = compiledBinaryVecVec(dtype_, dtype_, outputDtype_);
    break;
  case BinaryOpVariant::VecVecCmp:
    compiled = compiledBinaryVecVecCmp(dtype_, outputDtype_);
    break;
  case BinaryOpVariant::VecScalar:
    compiled = compiledBinaryVecScalar(dtype_, outputDtype_);
    break;
  case BinaryOpVariant::VecScalarCmp:
    compiled = compiledBinaryVecScalarCmp(dtype_, outputDtype_);
    break;
  case BinaryOpVariant::VecScalarBuf:
    compiled = compiledBinaryVecScalarBuf(dtype_, outputDtype_);
    break;
  case BinaryOpVariant::VecScalarBufCmp:
    compiled = compiledBinaryVecScalarBufCmp(dtype_, outputDtype_);
    break;
  }

  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> BinaryOpNode::outputShape() const {
  return store_->getTensor(inputs_[0]).getShape();
}

ThreadSize BinaryOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> BinaryOpNode::pushConstants() const {
  if (variant_ == BinaryOpVariant::VecScalar ||
      variant_ == BinaryOpVariant::VecScalarCmp) {
    // VecScalar/VecScalarCmp need both numElements and scalarBits
    struct PushConstants {
      uint32_t numElements;
      uint32_t scalarBits;
    } pc{static_cast<uint32_t>(numElements_), scalarBits_};
    return toBytes(pc);
  } else {
    // All other variants only need numElements
    uint32_t n = static_cast<uint32_t>(numElements_);
    return toBytes(n);
  }
}

} // namespace cut
