#include "TransposeOp.h"
#include "TensorStore.h"

namespace cut {

TransposeOpNode::TransposeOpNode(TensorStore &store,
                                 const Tensor &a,
                                 std::optional<uint32_t> spec)
    : OpNode(Transpose, store, spec) {
  const auto &buf = store.getTensor(a);
  dtype_ = buf.getDtype();
  auto shape = buf.getShape();
  if (shape.size() != 2) {
    throw std::runtime_error("Transpose requires a 2D tensor");
  }
  M_ = shape[0];
  N_ = shape[1];
  spec_ = spec.value_or(kTransposeDefaultVariant);
  inputs_ = {a};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType TransposeOpNode::shaderDtype() const {
  return dtype_;
}

DataType TransposeOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> TransposeOpNode::shader() const {
  return getCompiledTranspose(*spec_, dtype_, dtype_);
}

std::vector<uint32_t> TransposeOpNode::outputShape() const {
  return {N_, M_};
}

ThreadSize TransposeOpNode::dispatchSize() const {
  const auto &info = kTransposeVariants[*spec_];
  if (*spec_ == 0) {
    // Naive variant: original dispatch logic (vec4 output writes)
    uint32_t alignedM = (M_ + 3) & ~3u;
    uint32_t gridX = ((N_ + info.wgX - 1) / info.wgX) * info.wgX;
    uint32_t gridY = ((alignedM / 4 + info.wgY - 1) / info.wgY) * info.wgY;
    return {gridX, gridY, 1};
  }
  // Tiled variants: one thread per element in tile
  uint32_t gridX = ((N_ + info.effTileN - 1) / info.effTileN) * info.wgX;
  uint32_t gridY = ((M_ + info.effTileM - 1) / info.effTileM) * info.wgY;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> TransposeOpNode::pushConstants() const {
  uint32_t strideIn = (N_ + 3) & ~3u;
  uint32_t strideOut = (M_ + 3) & ~3u;
  struct PushConstants {
    uint32_t M, N, strideIn, strideOut;
  } pc{M_, N_, strideIn, strideOut};
  return toBytes(pc);
}

} // namespace cut
