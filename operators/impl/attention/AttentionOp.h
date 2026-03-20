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

} // namespace cut
