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

size_t AttentionOpNode::shaderKey() const {
  // The shader is compiled per cache dtype (F16 vs F32 cache), which is
  // not the output dtype — mix it in to avoid pipeline-cache collisions.
  return OpNode::shaderKey() | ((static_cast<size_t>(dtype_) & 0xF) << 56);
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

size_t FusedAttentionOpNode::shaderKey() const {
  // Shader compiled per cache dtype (see AttentionOpNode::shaderKey).
  return OpNode::shaderKey() | ((static_cast<size_t>(dtype_) & 0xF) << 56);
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

size_t BatchedFusedAttentionOpNode::shaderKey() const {
  // Shader compiled per cache dtype (see AttentionOpNode::shaderKey).
  return OpNode::shaderKey() | ((static_cast<size_t>(dtype_) & 0xF) << 56);
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

// --- BatchedKVCacheWriteOpNode ---

BatchedKVCacheWriteOpNode::BatchedKVCacheWriteOpNode(
    TensorStore &store,
    const Tensor &k,
    const Tensor &v,
    const Tensor &kCache,
    const Tensor &vCache,
    const Tensor &positions,
    const Tensor &cosTable,
    const Tensor &sinTable,
    uint32_t batchSize,
    uint32_t nKvHeads,
    uint32_t headDim,
    uint32_t kStride,
    uint32_t vStride,
    uint32_t kOffset,
    uint32_t vOffset,
    std::optional<uint32_t> spec)
    : OpNode(CacheWrite, store, spec) {
  dtype_ = store.getTensor(kCache).getDtype();
  batchSize_ = batchSize;
  headDim_ = headDim;
  halfDim_ = headDim / 2;
  kvDim_ = nKvHeads * headDim;
  alignedKvDim_ = (kvDim_ + 3) & ~static_cast<uint32_t>(3);
  kStride_ = kStride;
  vStride_ = vStride;
  kOffset_ = kOffset;
  vOffset_ = vOffset;

  inputs_ = {k, v, kCache, vCache, positions, cosTable, sinTable};
  // No fresh output: writes into the supplied caches. Use kCache as the
  // "output" handle for the runtime's barrier tracker — both kCache and
  // vCache get RWStructuredBuffer access in the shader, so any subsequent
  // dispatch reading either of them will see the writes.
  output_ = kCache;
}

DataType BatchedKVCacheWriteOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>>
BatchedKVCacheWriteOpNode::shader() const {
  return compiledBatchedKVCacheWrite(dtype_, dtype_);
}

size_t BatchedKVCacheWriteOpNode::shaderKey() const {
  // Avoid collision with CacheWriteOpNode (same OperatorEnum::CacheWrite).
  return OpNode::shaderKey() | (size_t{1} << 33);
}

std::vector<uint32_t> BatchedKVCacheWriteOpNode::outputShape() const {
  return store_->getTensor(output_).getShape();
}

ThreadSize BatchedKVCacheWriteOpNode::dispatchSize() const {
  // One workgroup of 256 threads per token; threads collaborate on writing
  // the row (K with RoPE + V).
  return {256, batchSize_, 1};
}

std::vector<uint8_t> BatchedKVCacheWriteOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t batchSize;
    uint32_t kvDim;
    uint32_t alignedKvDim;
    uint32_t headDim;
    uint32_t halfDim;
    uint32_t kStride;
    uint32_t vStride;
    uint32_t kOffset;
    uint32_t vOffset;
  } pc{batchSize_,    kvDim_,    alignedKvDim_, headDim_, halfDim_,
       kStride_,      vStride_,  kOffset_,      vOffset_};
  return toBytes(pc);
}

// --- BatchedAttentionReadCacheOpNode ---

BatchedAttentionReadCacheOpNode::BatchedAttentionReadCacheOpNode(
    TensorStore &store,
    const Tensor &q,
    const Tensor &kCache,
    const Tensor &vCache,
    const Tensor &positions,
    const Tensor &cosTable,
    const Tensor &sinTable,
    uint32_t batchSize,
    uint32_t nHeads,
    uint32_t nKvHeads,
    uint32_t headDim,
    uint32_t qStride,
    uint32_t qOffset,
    const Tensor &preallocOutput,
    std::optional<uint32_t> spec)
    : OpNode(Attention, store, spec) {
  dtype_ = store.getTensor(kCache).getDtype();
  batchSize_ = batchSize;
  nHeads_ = nHeads;
  nKvHeads_ = nKvHeads;
  headDim_ = headDim;
  alignedKvDim_ = ((nKvHeads * headDim) + 3) & ~static_cast<uint32_t>(3);
  nRep_ = nHeads / nKvHeads;
  scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));
  halfDim_ = headDim / 2;
  qStride_ = qStride;
  qOffset_ = qOffset;

  outShape_ = {batchSize, nHeads * headDim};
  auto outDtype = store.getTensor(q).getDtype();
  inputs_ = {q, kCache, vCache, positions, cosTable, sinTable};
  output_ = preallocOutput ? preallocOutput
                           : store.createTensorEmpty(outShape_, outDtype);
}

DataType BatchedAttentionReadCacheOpNode::outputDtype() const {
  return store_->getTensor(output_).getDtype();
}

std::optional<std::vector<uint32_t>>
BatchedAttentionReadCacheOpNode::shader() const {
  return compiledBatchedAttentionReadCache(dtype_, dtype_);
}

size_t BatchedAttentionReadCacheOpNode::shaderKey() const {
  // Avoid collision with AttentionOpNode (same OperatorEnum::Attention).
  // Also mix in the cache dtype the shader was compiled for (see
  // AttentionOpNode::shaderKey).
  return OpNode::shaderKey() | (size_t{1} << 34) |
         ((static_cast<size_t>(dtype_) & 0xF) << 56);
}

std::vector<uint32_t> BatchedAttentionReadCacheOpNode::outputShape() const {
  return outShape_;
}

ThreadSize BatchedAttentionReadCacheOpNode::dispatchSize() const {
  // X = nHeads workgroups, Y = batchSize workgroups. Each WG has 256 threads.
  return {nHeads_ * 256, batchSize_, 1};
}

std::vector<uint8_t> BatchedAttentionReadCacheOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t batchSize;
    uint32_t nHeads;
    uint32_t nKvHeads;
    uint32_t headDim;
    uint32_t alignedKvDim;
    uint32_t nRep;
    float scale;
    uint32_t halfDim;
    uint32_t qStride;
    uint32_t qOffset;
  } pc{batchSize_,    nHeads_,  nKvHeads_, headDim_, alignedKvDim_,
       nRep_,         scale_,   halfDim_,  qStride_, qOffset_};
  return toBytes(pc);
}

// ============================================================================
// DiTAttention: fused non-causal multi-head attention (no cache, no RoPE)
// ============================================================================

DiTAttentionOpNode::DiTAttentionOpNode(TensorStore &store,
                                       const Tensor &q,
                                       const Tensor &k,
                                       const Tensor &v,
                                       uint32_t nHeads,
                                       uint32_t headDim,
                                       float scale,
                                       std::optional<uint32_t> spec)
    : OpNode(Attention, store, spec) {
  const auto &qBuf = store.getTensor(q);
  const auto &kBuf = store.getTensor(k);
  const auto &vBuf = store.getTensor(v);

  dtype_ = qBuf.getDtype();
  if (kBuf.getDtype() != dtype_ || vBuf.getDtype() != dtype_ ||
      (dtype_ != DataType::Float32 && dtype_ != DataType::Float16)) {
    throw std::runtime_error(
        "DiTAttention requires q/k/v to share dtype Float32 or Float16");
  }

  const auto qShape = qBuf.getShape();
  const auto kShape = kBuf.getShape();
  const auto vShape = vBuf.getShape();
  if (qShape.size() != 2 || kShape.size() != 2 || vShape.size() != 2) {
    throw std::runtime_error("DiTAttention requires 2D inputs");
  }

  const uint32_t qCols = qShape[1];
  if (qCols != kShape[1] || qCols != vShape[1] || qCols != nHeads * headDim) {
    throw std::runtime_error(
        "DiTAttention requires Q/K/V columns == nHeads * headDim");
  }
  if (kShape[0] != vShape[0]) {
    throw std::runtime_error("DiTAttention requires K and V row counts to match");
  }
  if (headDim > 128 || headDim % 4 != 0) {
    throw std::runtime_error(
        "DiTAttention requires headDim <= 128 and a multiple of 4 "
        "(vectorized float4 global access)");
  }

  sq_ = qShape[0];
  skv_ = kShape[0];
  nHeads_ = nHeads;
  headDim_ = headDim;
  strideQ_ = (qCols + 3) & ~3u;
  strideKV_ = (kShape[1] + 3) & ~3u;
  strideO_ = strideQ_;
  scale_ = scale;

  outShape_ = {sq_, nHeads * headDim};
  inputs_ = {q, k, v};
  output_ = store.createTensorEmpty(outShape_, DataType::Float32);
}

DataType DiTAttentionOpNode::outputDtype() const { return DataType::Float32; }

std::optional<std::vector<uint32_t>> DiTAttentionOpNode::shader() const {
  return compiledDiTAttention(dtype_, DataType::Float32);
}

size_t DiTAttentionOpNode::shaderKey() const {
  // Bits 33/34 are taken by BatchedKVCacheWrite / BatchedAttentionReadCache
  // (all these nodes share OperatorEnum Attention). Mix in the input dtype
  // the shader was compiled for (see AttentionOpNode::shaderKey).
  return OpNode::shaderKey() | (size_t{1} << 35) |
         ((static_cast<size_t>(dtype_) & 0xF) << 56);
}

std::vector<uint32_t> DiTAttentionOpNode::outputShape() const {
  return outShape_;
}

ThreadSize DiTAttentionOpNode::dispatchSize() const {
  // X = ceil(sq / 16) workgroups x 256 threads (16 query rows per workgroup),
  // Y = nHeads workgroups. Same-head workgroups are adjacent in X so each
  // head's K/V stays L2-resident.
  return {((sq_ + 15) / 16) * 256, nHeads_, 1};
}

std::vector<uint8_t> DiTAttentionOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t sq;
    uint32_t skv;
    uint32_t nHeads;
    uint32_t headDim;
    uint32_t strideQ;
    uint32_t strideKV;
    uint32_t strideO;
    float scale;
  } pc{sq_, skv_, nHeads_, headDim_, strideQ_, strideKV_, strideO_, scale_};
  return toBytes(pc);
}

// --- FusedRoPEKVWriteOpNode ---

FusedRoPEKVWriteOpNode::FusedRoPEKVWriteOpNode(TensorStore &store,
                                               const Tensor &q,
                                               const Tensor &k,
                                               const Tensor &v,
                                               const Tensor &runtimeParams,
                                               const Tensor &cosTable,
                                               const Tensor &sinTable,
                                               const Tensor &kCache,
                                               const Tensor &vCache,
                                               uint32_t headDim,
                                               std::optional<uint32_t> spec)
    : OpNode(CacheWrite, store, spec) {
  cacheDtype_ = store.getTensor(kCache).getDtype();
  qDim_ = store.getTensor(q).getShape()[0];
  kvDim_ = store.getTensor(k).getShape()[0];
  headDim_ = headDim;
  halfDim_ = headDim / 2;
  alignedKvDim_ = (kvDim_ + 3) & ~static_cast<uint32_t>(3);

  inputs_ = {q, k, v, runtimeParams, cosTable, sinTable, kCache, vCache};
  // Real output: the roped Q vector. The cache writes are ordered by the
  // inter-dispatch barrier the tracker emits for this output before the
  // subsequent attention dispatch (which also reads both caches).
  output_ = store.createTensorEmpty({qDim_}, DataType::Float32);
}

DataType FusedRoPEKVWriteOpNode::outputDtype() const {
  return DataType::Float32;
}

std::optional<std::vector<uint32_t>> FusedRoPEKVWriteOpNode::shader() const {
  return compiledFusedRoPEKVWrite(cacheDtype_, cacheDtype_);
}

size_t FusedRoPEKVWriteOpNode::shaderKey() const {
  // Avoid collision with CacheWriteOpNode / BatchedKVCacheWriteOpNode
  // (same OperatorEnum::CacheWrite; BatchedKVCacheWrite salts bit 33).
  return OpNode::shaderKey() | (size_t{1} << 34);
}

std::vector<uint32_t> FusedRoPEKVWriteOpNode::outputShape() const {
  return {qDim_};
}

ThreadSize FusedRoPEKVWriteOpNode::dispatchSize() const {
  uint32_t totalWork = qDim_ + 2 * kvDim_;
  uint32_t gridX = ((totalWork + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> FusedRoPEKVWriteOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t qDim;
    uint32_t kvDim;
    uint32_t headDim;
    uint32_t halfDim;
    uint32_t alignedKvDim;
  } pc{qDim_, kvDim_, headDim_, halfDim_, alignedKvDim_};
  return toBytes(pc);
}

} // namespace cut
