#include "CastOp.h"
#include "CastShaders.generated.h"
#include "TensorStore.h"

namespace cut {

CastOpNode::CastOpNode(TensorStore &store,
                       const Tensor &input,
                       DataType targetDtype,
                       std::optional<uint32_t> spec)
    : OpNode(Cast, store, spec) {
  const auto &buf = store.getTensor(input);
  shape_ = buf.getShape();
  srcDtype_ = buf.getDtype();
  dstDtype_ = targetDtype;
  actualInner_ = shape_.empty() ? 1 : shape_.back();
  alignedInner_ = (actualInner_ + 3) & ~static_cast<uint32_t>(3);
  totalElements_ = static_cast<uint32_t>(actualElementCount(shape_));
  inputs_ = {input};
  output_ = store.createTensorEmpty(shape_, dstDtype_);
}

DataType CastOpNode::shaderDtype() const {
  return srcDtype_;
}

DataType CastOpNode::outputDtype() const {
  return dstDtype_;
}

size_t CastOpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(srcDtype_) & 0xF) << 16;
  key |= (static_cast<size_t>(dstDtype_) & 0xF) << 20;
  return key;
}

std::optional<std::vector<uint32_t>> CastOpNode::shader() const {
  return compiledCast(srcDtype_, dstDtype_);
}

std::vector<uint32_t> CastOpNode::outputShape() const {
  return shape_;
}

ThreadSize CastOpNode::dispatchSize() const {
  return {totalElements_, 1, 1};
}

std::vector<uint8_t> CastOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t alignedInner;
    uint32_t actualInner;
    uint32_t totalElements;
  } pc{alignedInner_, actualInner_, totalElements_};
  return toBytes(pc);
}

std::string CastOpNode::displayName() const {
  return "Cast";
}

} // namespace cut
