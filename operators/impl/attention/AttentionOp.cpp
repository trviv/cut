#include "AttentionOp.h"
#include "AttentionShaders.generated.h"
#include "TensorStore.h"

#include <cmath>
#include <stdexcept>

namespace cut {

// --- CacheWriteOpNode ---

CacheWriteOpNode::CacheWriteOpNode(TensorStore &store,
                                   const Tensor &newData,
                                   const Tensor &runtimeParams,
                                   const Tensor &cache,
                                   std::optional<uint32_t> spec)
    : OpNode(CacheWrite, store, spec) {
  const auto &cacheBuf = store.getTensor(cache);
  dtype_ = cacheBuf.getDtype();

  auto cacheShape = cacheBuf.getShape();
  kvDim_ = cacheShape.back();
  alignedKvDim_ = (kvDim_ + 3) & ~static_cast<uint32_t>(3);

  inputs_ = {newData, runtimeParams};
  // In-place: output IS the cache buffer
  output_ = cache;
}

DataType CacheWriteOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> CacheWriteOpNode::shader() const {
  return compiledCacheWrite(dtype_, dtype_);
}

std::vector<uint32_t> CacheWriteOpNode::outputShape() const {
  return store_->getTensor(output_).getShape();
}

ThreadSize CacheWriteOpNode::dispatchSize() const {
  uint32_t gridX = ((kvDim_ + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> CacheWriteOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t kvDim;
    uint32_t alignedKvDim;
  } pc{kvDim_, alignedKvDim_};
  return toBytes(pc);
}

// --- AttentionOpNode ---

AttentionOpNode::AttentionOpNode(TensorStore &store,
                                 const Tensor &q,
                                 const Tensor &kCache,
                                 const Tensor &vCache,
                                 const Tensor &runtimeParams,
                                 uint32_t nHeads,
                                 uint32_t nKvHeads,
                                 uint32_t headDim,
                                 const Tensor &preallocOutput,
                                 std::optional<uint32_t> spec)
    : OpNode(Attention, store, spec) {
  dtype_ = store.getTensor(q).getDtype();
  nHeads_ = nHeads;
  nKvHeads_ = nKvHeads;
  headDim_ = headDim;
  kvDim_ = nKvHeads * headDim;
  alignedKvDim_ = (kvDim_ + 3) & ~static_cast<uint32_t>(3);
  nRep_ = nHeads / nKvHeads;
  scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));

  outShape_ = {nHeads * headDim};

  inputs_ = {q, kCache, vCache, runtimeParams};
  output_ = preallocOutput ? preallocOutput
                           : store.createTensorEmpty(outShape_, dtype_);
}

DataType AttentionOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> AttentionOpNode::shader() const {
  return compiledAttention(dtype_, dtype_);
}

std::vector<uint32_t> AttentionOpNode::outputShape() const {
  return outShape_;
}

ThreadSize AttentionOpNode::dispatchSize() const {
  // One workgroup of 256 threads per attention head
  return {nHeads_ * 256, 1, 1};
}

std::vector<uint8_t> AttentionOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t nHeads;
    uint32_t nKvHeads;
    uint32_t headDim;
    uint32_t kvDim;
    uint32_t alignedKvDim;
    uint32_t nRep;
    float scale;
  } pc{nHeads_, nKvHeads_, headDim_, kvDim_, alignedKvDim_, nRep_, scale_};
  return toBytes(pc);
}

} // namespace cut
