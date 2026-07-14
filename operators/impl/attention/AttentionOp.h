#pragma once

#include "OpNode.h"

namespace cut {

/// Writes a 1D vector into a specific row of a 2D cache buffer (in-place).
/// Must be dispatched directly (not graph-recorded) since it modifies
/// the cache buffer in place.
class CacheWriteOpNode : public OpNode {
public:
  CacheWriteOpNode(TensorStore &store,
                   const Tensor &newData,
                   const Tensor &runtimeParams,
                   const Tensor &cache,
                   std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t kvDim_;
  uint32_t alignedKvDim_;
};

/// Computes scaled dot-product attention with GQA (grouped query attention).
/// Takes Q [nHeads * headDim], K_cache [maxSeqLen, kvDim],
/// V_cache [maxSeqLen, kvDim] and produces output [nHeads * headDim].
class AttentionOpNode : public OpNode {
public:
  AttentionOpNode(TensorStore &store,
                  const Tensor &q,
                  const Tensor &kCache,
                  const Tensor &vCache,
                  const Tensor &runtimeParams,
                  uint32_t nHeads,
                  uint32_t nKvHeads,
                  uint32_t headDim,
                  const Tensor &preallocOutput = {},
                  std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t nHeads_;
  uint32_t nKvHeads_;
  uint32_t headDim_;
  uint32_t kvDim_;
  uint32_t alignedKvDim_;
  uint32_t nRep_;
  float scale_;
  std::vector<uint32_t> outShape_;
};

/// Fused RoPE + CacheWrite + Attention in a single dispatch.
/// Saves 4 dispatches per layer (2 RoPE + 2 CacheWrite).
/// Inputs: q [nHeads*headDim], k [kvDim], v [kvDim],
///         kCache, vCache, runtimeParams, cosTable, sinTable.
class FusedAttentionOpNode : public OpNode {
public:
  FusedAttentionOpNode(TensorStore &store,
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
                       const Tensor &preallocOutput = {},
                       std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t nHeads_;
  uint32_t nKvHeads_;
  uint32_t headDim_;
  uint32_t kvDim_;
  uint32_t alignedKvDim_;
  uint32_t nRep_;
  float scale_;
  std::vector<uint32_t> outShape_;
};

/// Batched fused attention for prefill: processes N tokens in one dispatch.
/// Inputs: q [N, qStride], k [N, kStride], v [N, vStride],
///         kCache, vCache, posBuffer [N], cosTable, sinTable.
/// Output: [N, nHeads * headDim].
/// Batched K/V cache write for prefill: writes N tokens' K/V to cache
/// positions read from `positions` ([N] uint buffer), applying RoPE to K.
/// Designed to be paired with BatchedAttentionReadCacheOpNode — the Vulkan
/// barrier between the two dispatches ensures cache writes are visible to
/// the attention reads (which fixes the cross-workgroup race in the
/// single-dispatch BatchedFusedAttention).
class BatchedKVCacheWriteOpNode : public OpNode {
public:
  BatchedKVCacheWriteOpNode(TensorStore &store,
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
                            std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t batchSize_;
  uint32_t kvDim_;
  uint32_t alignedKvDim_;
  uint32_t headDim_;
  uint32_t halfDim_;
  uint32_t kStride_, vStride_;
  uint32_t kOffset_, vOffset_;
};

/// Batched attention for prefill, reading from a pre-populated K/V cache.
/// Pair with BatchedKVCacheWriteOpNode. Applies RoPE to Q inline.
class BatchedAttentionReadCacheOpNode : public OpNode {
public:
  BatchedAttentionReadCacheOpNode(TensorStore &store,
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
                                  const Tensor &preallocOutput = {},
                                  std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t batchSize_;
  uint32_t nHeads_;
  uint32_t nKvHeads_;
  uint32_t headDim_;
  uint32_t alignedKvDim_;
  uint32_t nRep_;
  float scale_;
  uint32_t halfDim_;
  uint32_t qStride_, qOffset_;
  std::vector<uint32_t> outShape_;
};

class BatchedFusedAttentionOpNode : public OpNode {
public:
  BatchedFusedAttentionOpNode(TensorStore &store,
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
                              const Tensor &preallocOutput = {},
                              std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t batchSize_;
  uint32_t nHeads_;
  uint32_t nKvHeads_;
  uint32_t headDim_;
  uint32_t kvDim_;
  uint32_t alignedKvDim_;
  uint32_t nRep_;
  float scale_;
  uint32_t qStride_, kStride_, vStride_;
  uint32_t qOffset_, kOffset_, vOffset_;
  std::vector<uint32_t> outShape_;
};

/// Fused non-causal multi-head attention for DiT models (no cache, no RoPE,
/// no masking, nKvHeads == nHeads). One workgroup per (query row, head) with
/// online softmax — no [sq, skv] score matrix is materialized.
class DiTAttentionOpNode : public OpNode {
public:
  DiTAttentionOpNode(TensorStore &store,
                     const Tensor &q,
                     const Tensor &k,
                     const Tensor &v,
                     uint32_t nHeads,
                     uint32_t headDim,
                     float scale,
                     std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  uint32_t sq_, skv_, nHeads_, headDim_;
  uint32_t strideQ_, strideKV_, strideO_;
  float scale_;
  std::vector<uint32_t> outShape_;
};

} // namespace cut
