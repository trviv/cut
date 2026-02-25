#include "BinaryOp.h"
#include "BinaryShaders.generated.h"
#include "Runtime.h"
#include "Shaders.h"

namespace cut {

BinaryOpNode::BinaryOpNode(OperatorEnum op,
                           Runtime &runtime,
                           const Tensor &a,
                           const TensorLike &b,
                           std::optional<uint32_t> spec)
    : OpNode(op, runtime, spec) {
  const auto &bufA = runtime.getTensor(a);
  const auto shapeA = bufA.getShape();
  dtype_ = bufA.getDtype();
  numElements_ = alignedElementCount(shapeA);

  // Detect variant based on b's type
  if (b.isHandle()) {
    const auto &bufB = runtime.getTensor(b.getHandle());
    const auto shapeB = bufB.getShape();

    if (actualElementCount(shapeB) == 1) {
      // Scalar buffer (shape {1})
      variant_ = BinaryOpVariant::VecScalarBuf;
      inputs_ = {a, b.getHandle()};
    } else {
      // Full tensor-tensor
      variant_ = BinaryOpVariant::VecVec;
      inputs_ = {a, b.getHandle()};

      // Validate shapes match
      if (actualElementCount(shapeA) != actualElementCount(shapeB)) {
        throw std::runtime_error(
            "Size mismatch: " + std::to_string(actualElementCount(shapeA)) +
            " vs " + std::to_string(actualElementCount(shapeB)));
      }
    }
  } else if (b.isScalar()) {
    // Inline scalar via push constant
    variant_ = BinaryOpVariant::VecScalar;
    scalarBits_ = b.getScalar<uint32_t>();
    inputs_ = {a};
  } else if (b.isData()) {
    // DataReference scalar via push constant
    variant_ = BinaryOpVariant::VecScalar;
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

  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
}

DataType BinaryOpNode::shaderDtype() const {
  return dtype_;
}

size_t BinaryOpNode::shaderKey() const {
  // Include variant in the key so VecVec, VecScalar, and VecScalarBuf
  // don't collide in the shader cache.
  return OpNode::shaderKey() ^ (static_cast<size_t>(variant_) << 48);
}

std::optional<std::vector<uint32_t>> BinaryOpNode::shader() const {
  std::optional<std::vector<uint32_t>> compiled;

  switch (variant_) {
  case BinaryOpVariant::VecVec:
    compiled = compiledBinaryVecVec(dtype_);
    break;
  case BinaryOpVariant::VecScalar:
    compiled = compiledBinaryVecScalar(dtype_);
    break;
  case BinaryOpVariant::VecScalarBuf:
    compiled = compiledBinaryVecScalarBuf(dtype_);
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
  return runtime_->getTensor(inputs_[0]).getShape();
}

ThreadSize BinaryOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> BinaryOpNode::pushConstants() const {
  if (variant_ == BinaryOpVariant::VecScalar) {
    // VecScalar needs both numElements and scalarBits
    struct PushConstants {
      uint32_t numElements;
      uint32_t scalarBits;
    } pc{static_cast<uint32_t>(numElements_), scalarBits_};
    return toBytes(pc);
  } else {
    // VecVec and VecScalarBuf only need numElements
    uint32_t n = static_cast<uint32_t>(numElements_);
    return toBytes(n);
  }
}

} // namespace cut
