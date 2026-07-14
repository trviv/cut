#include "RoPEOp.h"
#include "RopeShaders.generated.h"
#include "TensorStore.h"

namespace cut {

RoPEOpNode::RoPEOpNode(TensorStore &store,
                       const Tensor &x,
                       const Tensor &cosTable,
                       const Tensor &sinTable,
                       const Tensor &runtimeParams,
                       uint32_t headDim,
                       const Tensor &preallocOutput,
                       std::optional<uint32_t> spec)
    : OpNode(RoPE, store, spec) {
  const auto &buf = store.getTensor(x);
  dtype_ = buf.getDtype();
  outShape_ = buf.getShape();
  numElements_ = static_cast<uint32_t>(actualElementCount(outShape_));
  headDim_ = headDim;
  halfDim_ = headDim / 2;
  inputs_ = {x, cosTable, sinTable, runtimeParams};
  output_ = preallocOutput ? preallocOutput
                           : store.createTensorEmpty(outShape_, dtype_);
}

DataType RoPEOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> RoPEOpNode::shader() const {
  return compiledRoPE(dtype_, dtype_);
}

std::vector<uint32_t> RoPEOpNode::outputShape() const {
  return outShape_;
}

ThreadSize RoPEOpNode::dispatchSize() const {
  uint32_t gridX = ((numElements_ + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> RoPEOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numElements;
    uint32_t headDim;
    uint32_t halfDim;
  } pc{numElements_, headDim_, halfDim_};
  return toBytes(pc);
}

// --- BatchedRoPEOpNode ---

BatchedRoPEOpNode::BatchedRoPEOpNode(TensorStore &store,
                                     const Tensor &x,
                                     const Tensor &cosTable,
                                     const Tensor &sinTable,
                                     const Tensor &positions,
                                     uint32_t batchSize,
                                     uint32_t dim,
                                     uint32_t inRowStride,
                                     uint32_t inRowOffset,
                                     uint32_t headDim,
                                     const Tensor &preallocOutput,
                                     std::optional<uint32_t> spec)
    : OpNode(RoPE, store, spec) {
  const auto &buf = store.getTensor(x);
  dtype_ = buf.getDtype();
  batchSize_ = batchSize;
  dim_ = dim;
  alignedDim_ = (dim + 3) & ~3u;
  inRowStride_ = inRowStride;
  inRowOffset_ = inRowOffset;
  headDim_ = headDim;
  halfDim_ = headDim / 2;
  outShape_ = (batchSize == 1) ? std::vector<uint32_t>{dim}
                                : std::vector<uint32_t>{batchSize, dim};
  inputs_ = {x, cosTable, sinTable, positions};
  output_ = preallocOutput ? preallocOutput
                           : store.createTensorEmpty(outShape_, dtype_);
}

DataType BatchedRoPEOpNode::outputDtype() const {
  return dtype_;
}

size_t BatchedRoPEOpNode::shaderKey() const {
  // Toggle a high bit so we don't collide with RoPEOpNode in the pipeline
  // cache (both ops share OperatorEnum::RoPE).
  return OpNode::shaderKey() | (size_t{1} << 32);
}

std::optional<std::vector<uint32_t>> BatchedRoPEOpNode::shader() const {
  return compiledBatchedRoPE(dtype_, dtype_);
}

std::vector<uint32_t> BatchedRoPEOpNode::outputShape() const {
  return outShape_;
}

ThreadSize BatchedRoPEOpNode::dispatchSize() const {
  // (ceil(dim/256)*256, batchSize, 1) — each Y-workgroup handles one token,
  // X dimension covers the dim-length vector in chunks of 256 threads.
  uint32_t gridX = ((dim_ + 255) / 256) * 256;
  return {gridX, batchSize_, 1};
}

std::vector<uint8_t> BatchedRoPEOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t batchSize;
    uint32_t dim;
    uint32_t alignedDim;
    uint32_t inRowStride;
    uint32_t inRowOffset;
    uint32_t headDim;
    uint32_t halfDim;
  } pc{batchSize_, dim_, alignedDim_, inRowStride_,
       inRowOffset_, headDim_, halfDim_};
  return toBytes(pc);
}

// --- RoPEInterleavedOpNode ---

RoPEInterleavedOpNode::RoPEInterleavedOpNode(TensorStore &store,
                                             const Tensor &x,
                                             const Tensor &cosTable,
                                             const Tensor &sinTable,
                                             std::optional<uint32_t> spec)
    : OpNode(RoPE, store, spec) {
  const auto &buf = store.getTensor(x);
  dtype_ = buf.getDtype();
  if (dtype_ != DataType::Float32) {
    throw std::runtime_error("applyRoPEInterleaved: only Float32 is supported");
  }
  outShape_ = buf.getShape();
  numElements_ = static_cast<uint32_t>(actualElementCount(outShape_));
  if (outShape_.back() % 4 != 0 || outShape_.back() < 2) {
    throw std::runtime_error(
        "applyRoPEInterleaved: innermost dim must be a multiple of 4");
  }
  auto cosElements = static_cast<uint32_t>(
      actualElementCount(store.getTensor(cosTable).getShape()));
  auto sinElements = static_cast<uint32_t>(
      actualElementCount(store.getTensor(sinTable).getShape()));
  if (cosElements != numElements_ || sinElements != numElements_) {
    throw std::runtime_error(
        "applyRoPEInterleaved: cosTable and sinTable must have same shape as x");
  }
  inputs_ = {x, cosTable, sinTable};
  output_ = store.createTensorEmpty(outShape_, dtype_);
}

DataType RoPEInterleavedOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> RoPEInterleavedOpNode::shader() const {
  return compiledRoPEInterleaved(dtype_, dtype_);
}

size_t RoPEInterleavedOpNode::shaderKey() const {
  return OpNode::shaderKey() | (size_t{1} << 35);
}

std::vector<uint32_t> RoPEInterleavedOpNode::outputShape() const {
  return outShape_;
}

ThreadSize RoPEInterleavedOpNode::dispatchSize() const {
  uint32_t gridX = ((numElements_ + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> RoPEInterleavedOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numElements;
  } pc{numElements_};
  return toBytes(pc);
}

} // namespace cut
