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
  // Shader variant selected by cache dtype (F16 cache = F16 shader with
  // mixed-precision: Q/output remain F32, only cache reads are F16).
  dtype_ = store.getTensor(kCache).getDtype();
  nHeads_ = nHeads;
  nKvHeads_ = nKvHeads;
  headDim_ = headDim;
  kvDim_ = nKvHeads * headDim;
  alignedKvDim_ = (kvDim_ + 3) & ~static_cast<uint32_t>(3);
  nRep_ = nHeads / nKvHeads;
  scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));

  outShape_ = {nHeads * headDim};

  // Output is always Float32 (attention accumulates in full precision)
  auto outDtype = store.getTensor(q).getDtype();
  inputs_ = {q, kCache, vCache, runtimeParams};
  output_ = preallocOutput ? preallocOutput
                           : store.createTensorEmpty(outShape_, outDtype);
}

DataType AttentionOpNode::outputDtype() const {
  // Output dtype matches Q (Float32), not cache dtype
  return store_->getTensor(output_).getDtype();
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

// --- FusedAttentionOpNode ---

FusedAttentionOpNode::FusedAttentionOpNode(TensorStore &store,
                                           const Tensor &q,
                                           const Tensor &k,
                                           const Tensor &v,
                                           const Tensor &kCache,
                                           const Tensor &vCache,
                                           const Tensor &runtimeParams,
                                           const Tensor &cosTable,
                                           const Tensor &sinTable,
                                           uint32_t nHeads,
                                           uint32_t nKvHeads,
                                           uint32_t headDim,
                                           const Tensor &preallocOutput,
                                           std::optional<uint32_t> spec)
    : OpNode(FusedAttention, store, spec) {
  // Shader variant selected by cache dtype (F16 cache = F16 shader).
  dtype_ = store.getTensor(kCache).getDtype();
  nHeads_ = nHeads;
  nKvHeads_ = nKvHeads;
  headDim_ = headDim;
  kvDim_ = nKvHeads * headDim;
  alignedKvDim_ = (kvDim_ + 3) & ~static_cast<uint32_t>(3);
  nRep_ = nHeads / nKvHeads;
  scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));

  outShape_ = {nHeads * headDim};

  // Output is always Float32 (attention accumulates in full precision)
  auto outDtype = store.getTensor(q).getDtype();
  // Order: q, k, v, kCache, vCache, runtimeParams, cosTable, sinTable
  inputs_ = {q, k, v, kCache, vCache, runtimeParams, cosTable, sinTable};
  output_ = preallocOutput ? preallocOutput
                           : store.createTensorEmpty(outShape_, outDtype);
}

DataType FusedAttentionOpNode::outputDtype() const {
  return store_->getTensor(output_).getDtype();
}

std::optional<std::vector<uint32_t>> FusedAttentionOpNode::shader() const {
  return compiledFusedAttention(dtype_, dtype_);
}

std::vector<uint32_t> FusedAttentionOpNode::outputShape() const {
  return outShape_;
}

ThreadSize FusedAttentionOpNode::dispatchSize() const {
  return {nHeads_ * 256, 1, 1};
}

std::vector<uint8_t> FusedAttentionOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t nHeads;
    uint32_t nKvHeads;
    uint32_t headDim;
    uint32_t kvDim;
    uint32_t alignedKvDim;
    uint32_t nRep;
    float scale;
    uint32_t halfDim; // headDim / 2 for RoPE
  } pc{nHeads_,       nKvHeads_, headDim_, kvDim_,
       alignedKvDim_, nRep_,     scale_,   headDim_ / 2};
  return toBytes(pc);
}

// --- BatchedFusedAttentionOpNode ---

BatchedFusedAttentionOpNode::BatchedFusedAttentionOpNode(
    TensorStore &store,
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor &kCache,
    const Tensor &vCache,
    const Tensor &posBuffer,
    const Tensor &cosTable,
    const Tensor &sinTable,
    uint32_t batchSize,
    uint32_t nHeads,
    uint32_t nKvHeads,
    uint32_t headDim,
    uint32_t qStride,
    uint32_t kStride,
    uint32_t vStride,
    uint32_t qOffset,
    uint32_t kOffset,
    uint32_t vOffset,
    const Tensor &preallocOutput,
    std::optional<uint32_t> spec)
    : OpNode(BatchedFusedAttention, store, spec) {
  dtype_ = store.getTensor(kCache).getDtype();
  batchSize_ = batchSize;
  nHeads_ = nHeads;
  nKvHeads_ = nKvHeads;
  headDim_ = headDim;
  kvDim_ = nKvHeads * headDim;
  alignedKvDim_ = (kvDim_ + 3) & ~static_cast<uint32_t>(3);
  nRep_ = nHeads / nKvHeads;
  scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));
  qStride_ = qStride;
  kStride_ = kStride;
  vStride_ = vStride;
  qOffset_ = qOffset;
  kOffset_ = kOffset;
  vOffset_ = vOffset;

  outShape_ = {batchSize, nHeads * headDim};

  auto outDtype = store.getTensor(q).getDtype();
  inputs_ = {q, k, v, kCache, vCache, posBuffer, cosTable, sinTable};
  output_ = preallocOutput ? preallocOutput
                           : store.createTensorEmpty(outShape_, outDtype);
}

DataType BatchedFusedAttentionOpNode::outputDtype() const {
  return store_->getTensor(output_).getDtype();
}

std::optional<std::vector<uint32_t>>
BatchedFusedAttentionOpNode::shader() const {
  return compiledBatchedFusedAttention(dtype_, dtype_);
}

std::vector<uint32_t> BatchedFusedAttentionOpNode::outputShape() const {
  return outShape_;
}

ThreadSize BatchedFusedAttentionOpNode::dispatchSize() const {
  // X = nHeads workgroups (one per head), Y = batchSize (one per token)
  return {nHeads_ * 256, batchSize_, 1};
}

std::vector<uint8_t> BatchedFusedAttentionOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t batchSize;
    uint32_t nHeads;
    uint32_t nKvHeads;
    uint32_t headDim;
    uint32_t kvDim;
    uint32_t alignedKvDim;
    uint32_t nRep;
    float scale;
    uint32_t halfDim;
    uint32_t qStride;
    uint32_t kStride;
    uint32_t vStride;
    uint32_t qOffset;
    uint32_t kOffset;
    uint32_t vOffset;
  } pc{batchSize_,    nHeads_,  nKvHeads_, headDim_,     kvDim_,
       alignedKvDim_, nRep_,    scale_,    headDim_ / 2, qStride_,
       kStride_,      vStride_, qOffset_,  kOffset_,     vOffset_};
  return toBytes(pc);
}

} // namespace cut
