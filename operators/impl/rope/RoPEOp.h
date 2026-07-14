#pragma once

#include "OpNode.h"

namespace cut {

class RoPEOpNode : public OpNode {
public:
  RoPEOpNode(TensorStore &store,
             const Tensor &x,
             const Tensor &cosTable,
             const Tensor &sinTable,
             const Tensor &runtimeParams,
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
  uint32_t numElements_;
  uint32_t headDim_;
  uint32_t halfDim_;
  std::vector<uint32_t> outShape_;
};

/// Batched RoPE — applies rotary position embedding to N tokens in one
/// dispatch. Reads a [N, dim] sub-block at column `inRowOffset` of a
/// [N, inRowStride] input buffer (so it can rotate Q or K columns of a
/// fused QKV matmul output without copying). Writes a fresh contiguous
/// [N, dim] output. Per-token positions come from a [N] uint buffer.
class BatchedRoPEOpNode : public OpNode {
public:
  BatchedRoPEOpNode(TensorStore &store,
                    const Tensor &x,
                    const Tensor &cosTable,
                    const Tensor &sinTable,
                    const Tensor &positions,
                    uint32_t batchSize,
                    uint32_t dim,
                    uint32_t inRowStride,
                    uint32_t inRowOffset,
                    uint32_t headDim,
                    const Tensor &preallocOutput = {},
                    std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  // BatchedRoPE shares OperatorEnum::RoPE with the per-token RoPE op but
  // uses a different shader and binding layout (5 inputs vs 4: positions
  // buffer is per-token instead of a single runtimeParams[0]). Without a
  // distinct shaderKey, the runtime's pipeline cache collides and one of
  // the two ops gets dispatched with the other's pipeline → wrong output.
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t batchSize_;
  uint32_t dim_;
  uint32_t alignedDim_;
  uint32_t inRowStride_;
  uint32_t inRowOffset_;
  uint32_t headDim_;
  uint32_t halfDim_;
  std::vector<uint32_t> outShape_;
};

/// Interleaved-pair RoPE (LTX/DiT convention): rotates consecutive element
/// pairs using cos/sin tables with the SAME shape as x (no position buffer,
/// no head dim). Requires the innermost dimension of x to be a multiple of 4
/// (keeps pairs away from alignment padding).
class RoPEInterleavedOpNode : public OpNode {
public:
  RoPEInterleavedOpNode(TensorStore &store,
                        const Tensor &x,
                        const Tensor &cosTable,
                        const Tensor &sinTable,
                        std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  // Shares OperatorEnum::RoPE with the other RoPE ops but uses a different
  // shader/binding layout — tag the key to avoid pipeline-cache collisions
  // (same pattern/reason as BatchedRoPEOpNode::shaderKey).
  size_t shaderKey() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;

private:
  DataType dtype_;
  uint32_t numElements_;
  std::vector<uint32_t> outShape_;
};

} // namespace cut
