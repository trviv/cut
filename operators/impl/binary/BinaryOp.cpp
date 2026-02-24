#include "BinaryOp.h"
#include "BinaryShaders.generated.h"
#include "Runtime.h"
#include "Shaders.h"

namespace cut {

// --- BinaryVecVecOpNode ---

BinaryVecVecOpNode::BinaryVecVecOpNode(OperatorEnum op,
                                       Runtime &runtime,
                                       const Tensor &a,
                                       const Tensor &b,
                                       std::optional<uint32_t> spec)
    : OpNode(op, runtime, spec) {
  const auto &bufA = runtime.getTensor(a);
  const auto &bufB = runtime.getTensor(b);
  const auto shapeA = bufA.getShape();
  const auto shapeB = bufB.getShape();
  dtype_ = bufA.getDtype();
  numElements_ = alignedElementCount(shapeA);
  if (actualElementCount(shapeA) != actualElementCount(shapeB)) {
    throw std::runtime_error(
        "Size mismatch: " + std::to_string(actualElementCount(shapeA)) +
        " vs " + std::to_string(actualElementCount(shapeB)));
  }
  inputs_ = {a, b};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType BinaryVecVecOpNode::shaderDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> BinaryVecVecOpNode::shader() const {
  auto compiled = compiledBinaryVecVec(dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> BinaryVecVecOpNode::outputShape() const {
  return runtime_->getTensor(inputs_[0]).getShape();
}

ThreadSize BinaryVecVecOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> BinaryVecVecOpNode::pushConstants() const {
  uint32_t n = static_cast<uint32_t>(numElements_);
  return toBytes(n);
}

// --- BinaryVecScalarOpNode ---

BinaryVecScalarOpNode::BinaryVecScalarOpNode(OperatorEnum op,
                                             Runtime &runtime,
                                             const Tensor &a,
                                             const ComputeData &b,
                                             std::optional<uint32_t> spec)
    : OpNode(op, runtime, spec) {
  const auto &buf = runtime.getTensor(a);
  dtype_ = buf.getDtype();
  if (b.isScalar()) {
    scalarBits_ = b.getScalar<uint32_t>();
    inputs_ = {a};
    usesHandleScalar_ = false;
  } else if (b.isHandle()) {
    inputs_ = {a, b.getHandle()};
    usesHandleScalar_ = true;
  } else {
    logErr("Second binary operator between vector and a scalar can only be a "
           "scalar or a compute handle.");
  }
  numElements_ = alignedElementCount(buf.getShape());
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType BinaryVecScalarOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> BinaryVecScalarOpNode::outputShape() const {
  return runtime_->getTensor(inputs_[0]).getShape();
}

ThreadSize BinaryVecScalarOpNode::dispatchSize() const {
  return {static_cast<uint32_t>(numElements_), 1, 1};
}

std::vector<uint8_t> BinaryVecScalarOpNode::pushConstants() const {
  if (usesHandleScalar_) {
    // VecScalarBuf shader only needs numElements
    struct PushConstants {
      uint32_t numElements;
    } pc{static_cast<uint32_t>(numElements_)};
    return toBytes(pc);
  } else {
    // VecScalar shader needs both numElements and scalarBits
    struct PushConstants {
      uint32_t numElements;
      uint32_t scalarBits;
    } pc{static_cast<uint32_t>(numElements_), scalarBits_};
    return toBytes(pc);
  }
}

std::optional<std::vector<uint32_t>> BinaryVecScalarOpNode::shader() const {
  std::optional<std::vector<uint32_t>> compiled;
  if (usesHandleScalar_) {
    compiled = compiledBinaryVecScalarBuf(dtype_);
  } else {
    compiled = compiledBinaryVecScalar(dtype_);
  }

  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    // Patch op_enum specialization constant (constant_id = 1) with the
    // actual operator value so the compiled shader executes the right op.
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }

  return std::nullopt;
}

} // namespace cut
