#pragma once

#include "ShapeUtils.h"
#include "graph/Graph.h"

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <ComputeOps.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace cut {

class MatMulOpNode;
class OpNode;
class Runtime;
class TensorStore;

/**
 * High-level tensor operations implemented in C++.
 * Works directly on Tensor objects. Uses TensorStore for tensor
 * metadata and creation, and Runtime for GPU dispatch.
 */
class Operations {
public:
  /// Constructs the Operations facade for one device of the runtime.
  explicit Operations(Runtime &runtime, size_t deviceId = 0);

  // ===== Generic element-wise ops =====

  Tensor binaryOp(OperatorEnum op,
                  const Tensor &a,
                  const TensorLike &b,
                  std::optional<uint32_t> spec = {});

  /// Row-broadcast binary op: out[r, c] = op(a[r, c], b[c]);
  /// a is [.., cols], b is a 1-D [cols] vector.
  Tensor binaryOpRowBcast(OperatorEnum op,
                          const Tensor &a,
                          const Tensor &b,
                          std::optional<uint32_t> spec = {});

  Tensor
  unaryOp(OperatorEnum op, const Tensor &a, std::optional<uint32_t> spec = {});

  // ===== Reduction ops =====

  Tensor reduce(OperatorEnum op,
                const Tensor &a,
                std::optional<int> dim = {},
                std::optional<uint32_t> spec = {});

  // ===== Matrix ops =====

  /// Standard matmul: C = A * B
  Tensor
  matmul(const Tensor &a, const Tensor &b, std::optional<uint32_t> spec = {});

  /// Quantized matmul: C = A * dequant(packedB, scales)
  /// Auto-detects Q4 vs Q8 from input shapes.
  Tensor matmul(const Tensor &a,
                const Tensor &packedB,
                const Tensor &scales,
                std::optional<uint32_t> spec = {});

  /// Standard matmul with unary fusion: unaryOp(A * B)
  Tensor matmulUnary(OperatorEnum unaryOp,
                     const Tensor &a,
                     const Tensor &b,
                     std::optional<uint32_t> spec = {});

  /// Quantized matmul with unary fusion: unaryOp(A * dequant(packedB, scales))
  Tensor matmulUnary(OperatorEnum unaryOp,
                     const Tensor &a,
                     const Tensor &packedB,
                     const Tensor &scales,
                     std::optional<uint32_t> spec = {});

  /// Standard matmul with binary op: binaryOp(A * B, D)
  /// Fuses the residual add into the matmul (saves one BinaryAdd dispatch
  /// + a full read/write of [M, N]).
  Tensor matmulBinary(OperatorEnum binaryOp,
                      const Tensor &a,
                      const Tensor &b,
                      const Tensor &d,
                      std::optional<uint32_t> spec = {});

  /// Quantized matmul with binary op: binaryOp(A * dequant(packedB, scales), D)
  Tensor matmulBinary(OperatorEnum binaryOp,
                      const Tensor &a,
                      const Tensor &packedB,
                      const Tensor &scales,
                      const Tensor &d,
                      std::optional<uint32_t> spec = {});

  Tensor transpose(const Tensor &a, std::optional<uint32_t> spec = {});

  Tensor
  dot(const Tensor &a, const Tensor &b, std::optional<uint32_t> spec = {});

  // ===== Special ops =====

  Tensor clamp(const Tensor &a,
               DataReference clampData,
               std::optional<uint32_t> spec = {});

  Tensor where(const Tensor &cond,
               const Tensor &x,
               const Tensor &y,
               std::optional<uint32_t> spec = {});

  // ===== Cumulative ops =====

  Tensor cumOp(const Tensor &a,
               OperatorEnum op,
               std::optional<int> dim = {},
               std::optional<uint32_t> spec = {});

  // ===== Statistical ops =====

  Tensor variance(const Tensor &a, int correction, std::optional<int> dim = {});

  /// Single-pass variance using Welford's algorithm (GPU shader).
  Tensor varianceFused(const Tensor &a,
                       int correction = 0,
                       std::optional<int> dim = {},
                       std::optional<uint32_t> spec = {});

  /// Standalone RMS: sqrt(mean(x^2))
  Tensor rms(const Tensor &a,
             std::optional<int> dim = {},
             std::optional<uint32_t> spec = {});

  /// Numerically stable log(sum(exp(x))) using online normalizer algorithm.
  Tensor logSumExp(const Tensor &a,
                   std::optional<int> dim = {},
                   std::optional<uint32_t> spec = {});

  // ===== Softmax =====

  Tensor softmax(const Tensor &a, int dim);

  Tensor logSoftmax(const Tensor &a, int dim);

  /// Fused single-kernel softmax (2-pass: online normalizer + normalize).
  Tensor
  softmaxFused(const Tensor &a, int dim, std::optional<uint32_t> spec = {});

  /// Fused single-kernel log-softmax.
  Tensor
  logSoftmaxFused(const Tensor &a, int dim, std::optional<uint32_t> spec = {});

  // ===== Tensor creation =====

  Tensor arange(DataReference start,
                DataReference end,
                DataReference step,
                DataType dtype);
  Tensor
  linspace(DataReference start, DataReference end, int steps, DataType dtype);

  Tensor full(const std::vector<uint32_t> &shape,
              DataReference fillValue,
              DataType dtype);

  // ===== Shape ops (copy data to new buffer with new shape) =====

  Tensor reshape(const Tensor &a, const std::vector<int32_t> &newShape);

  Tensor squeeze(const Tensor &a, std::optional<int> dim);

  Tensor unsqueeze(const Tensor &a, int dim);

  Tensor
  unflatten(const Tensor &a, int dim, const std::vector<uint32_t> &sizes);

  Tensor flatten(const Tensor &a, int startDim, int endDim);

  // ===== Norm =====

  Tensor norm(const Tensor &a,
              std::optional<int> dim = {},
              std::optional<uint32_t> spec = {});

  // ===== Prefix scan =====

  Tensor prefixScan(const Tensor &a,
                    OperatorEnum op,
                    std::optional<uint32_t> spec = {});

  // ===== Convolution ops =====

  Tensor conv1d(const Tensor &input,
                const Tensor &weight,
                uint32_t stride = 1,
                uint32_t padding = 0,
                std::optional<uint32_t> spec = {});

  Tensor conv2d(const Tensor &input,
                const Tensor &weight,
                uint32_t strideH = 1,
                uint32_t strideW = 1,
                uint32_t padH = 0,
                uint32_t padW = 0,
                std::optional<uint32_t> spec = {});

  // ===== Pooling ops =====

  Tensor maxPool2d(const Tensor &input,
                   uint32_t kernelH,
                   uint32_t kernelW,
                   uint32_t strideH = 1,
                   uint32_t strideW = 1,
                   uint32_t padH = 0,
                   uint32_t padW = 0,
                   std::optional<uint32_t> spec = {});

  Tensor avgPool2d(const Tensor &input,
                   uint32_t kernelH,
                   uint32_t kernelW,
                   uint32_t strideH = 1,
                   uint32_t strideW = 1,
                   uint32_t padH = 0,
                   uint32_t padW = 0,
                   std::optional<uint32_t> spec = {});

  Tensor adaptiveAvgPool2d(const Tensor &input,
                           uint32_t outH,
                           uint32_t outW,
                           std::optional<uint32_t> spec = {});

  // ===== Normalization ops =====

  Tensor layerNorm(const Tensor &input,
                   const std::vector<uint32_t> &normalizedShape,
                   const Tensor *weight = nullptr,
                   const Tensor *bias = nullptr,
                   float eps = 1e-5f);

  Tensor batchNorm(const Tensor &input,
                   const Tensor &runningMean,
                   const Tensor &runningVar,
                   const Tensor *weight = nullptr,
                   const Tensor *bias = nullptr,
                   float eps = 1e-5f);

  Tensor rmsNorm(const Tensor &x,
                 const Tensor &weight,
                 float eps = 1e-5f,
                 std::optional<uint32_t> spec = {});

  Tensor extendedRmsNorm(const Tensor &residual_base,
                         const Tensor &delta,
                         const Tensor &weight,
                         float eps = 1e-5f,
                         std::optional<uint32_t> spec = {});

  // ===== Embedding ops =====

  Tensor embedding(const Tensor &indices,
                   const Tensor &weight,
                   const Tensor &preallocOutput = {},
                   std::optional<uint32_t> spec = {});

  // ===== RoPE =====

  /// Batched RoPE for prefill: applies rotary embedding to N tokens with
  /// per-token positions read from a [N] uint buffer. Reads from a
  /// [N, inRowStride] input at column inRowOffset; writes a fresh
  /// contiguous [N, dim] output. inRowStride/inRowOffset = (dim, 0) for
  /// a plain [N, dim] input; for a fused QKV matmul output set
  /// inRowStride = qdim+2*kvdim and inRowOffset = 0 (for Q) or qdim (for K).
  Tensor applyBatchedRoPE(const Tensor &x,
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

  Tensor applyRoPE(const Tensor &x,
                   const Tensor &cosTable,
                   const Tensor &sinTable,
                   const Tensor &runtimeParams,
                   uint32_t headDim,
                   const Tensor &preallocOutput = {},
                   std::optional<uint32_t> spec = {});

  /// Interleaved-pair RoPE (LTX/DiT): out = x*cos + rotate_pairs(x)*sin,
  /// where rotate_pairs swaps each consecutive pair (x0,x1) -> (-x1,x0).
  /// cos/sin tables have the same shape as x. Innermost dim % 4 == 0.
  Tensor applyRoPEInterleaved(const Tensor &x,
                              const Tensor &cosTable,
                              const Tensor &sinTable,
                              std::optional<uint32_t> spec = {});

  // ===== Attention =====

  /// Write a 1D vector into a specific row of a 2D cache buffer (in-place).
  /// Dispatched immediately (not graph-recorded).
  void cacheWrite(const Tensor &cache,
                  const Tensor &newData,
                  const Tensor &runtimeParams);

  /// Scaled dot-product attention with GQA support.
  Tensor attention(const Tensor &q,
                   const Tensor &kCache,
                   const Tensor &vCache,
                   const Tensor &runtimeParams,
                   uint32_t nHeads,
                   uint32_t nKvHeads,
                   uint32_t headDim,
                   const Tensor &preallocOutput = {},
                   std::optional<uint32_t> spec = {});

  /// Fused RoPE + CacheWrite + Attention (saves 4 dispatches/layer).
  void fusedAttention(const Tensor &q,
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
                      const Tensor &preallocOutput = {});

  /// Batched K + V cache write for prefill: writes N tokens' K (with RoPE)
  /// and V to cache positions read from a [N] uint buffer. Pair with
  /// batchedAttentionReadCache. Splitting the old batchedFusedAttention
  /// into two dispatches fixes its cross-workgroup race.
  void batchedKVCacheWrite(const Tensor &k,
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

  /// Batched attention reading a pre-populated K/V cache (Q gets RoPE
  /// inline). Pair with batchedKVCacheWrite.
  Tensor batchedAttentionReadCache(const Tensor &q,
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

  /// Batched fused attention for prefill: N tokens in one dispatch.
  Tensor batchedFusedAttention(const Tensor &q,
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
                               const Tensor &preallocOutput = {});

  // ===== Expand (broadcast) =====

  Tensor expand(const Tensor &a,
                const std::vector<uint32_t> &targetShape,
                std::optional<uint32_t> spec = {});

  // ===== Slice (zero-copy view) =====

  /// Extracts a contiguous slice [start, end) along `dim` of a 1D tensor.
  /// Returns a zero-copy buffer view — no GPU dispatch.
  Tensor slice(const Tensor &a, uint32_t dim, uint32_t start, uint32_t end);

  // ===== Padding ops =====

  Tensor pad(const Tensor &input,
             const std::vector<uint32_t> &padWidths,
             float value = 0.0f,
             std::optional<uint32_t> spec = {});

  // ===== Repetition penalty =====

  /// GPU repetition penalty: applies conditional scaling to logits.
  /// logits and penaltyFactors must be same-shaped Float32 tensors.
  Tensor repetitionPenalty(const Tensor &logits, const Tensor &penaltyFactors);

  // ===== Q4 nibble transpose =====

  /// GPU Q4_0 nibble transpose: [N, K/2] -> [K, N/2] packed.
  /// Combines unpack, transpose, and repack in a single dispatch.
  Tensor transposeQ4(const Tensor &packedInput, uint32_t N, uint32_t K);

  // ===== Dequantization =====

  /// GPU dequantization: converts raw quantized bytes to Float32.
  /// Input: 1D Int8 tensor of raw GGUF block data.
  /// Output: 2D Float32 tensor [rows, cols].
  /// @param format DequantFormat enum value (BF16=0, Q4_K=1, Q5_K=2, Q6_K=3).
  Tensor dequantize(const Tensor &rawData,
                    uint32_t format,
                    uint32_t rows,
                    uint32_t cols);

  // ===== Type conversion =====

  /// Casts a tensor to a different dtype via a graph-recorded CastOpNode.
  /// Returns the input unchanged if it already has the target dtype.
  /// Results are cached per graph: casting the same tensor to the same dtype
  /// multiple times returns the same output tensor (no redundant graph nodes).
  Tensor cast(const Tensor &input, DataType targetDtype);

  // ===== Sort (in-place) =====

  void sortBitonic(const Tensor &keys,
                   const Tensor &vals,
                   std::optional<uint32_t> spec = {});

  void sortRadix(const Tensor &keys,
                 const Tensor &vals,
                 std::optional<uint32_t> spec = {});

  // ===== Profiling =====

  /// Enables or disables per-dispatch GPU profiling.
  void setProfilingEnabled(bool enabled);

  // ===== Direct dispatch =====

  void dispatch(std::unique_ptr<OpNode> node);
  void dispatch(OpNode &node);

  // ===== Graph =====

  /// Flush the internal graph: optimize, execute, and write results into
  /// placeholder buffers. No-op if the graph is empty or already executed.
  void flush();

  /// Insert an explicit compute-to-compute barrier in the command buffer.
  /// Used when buffer views (which share a parent buffer) need a barrier
  /// that the automatic dependency tracker can't infer.
  void barrier();

  /// Move the current graph out and replace with a fresh one.
  /// Used by GraphBuilder to obtain the recorded graph.
  std::unique_ptr<graph::Graph> takeGraph();

  /// Mark a tensor as a graph output (used by GraphBuilder).
  void markGraphOutput(const Tensor &t);

  /// Register a pre-existing GPU tensor as a graph input (InputOpNode).
  /// Used by GraphBuilder.
  Tensor registerInput(const Tensor &gpuHandle, bool isConstant = false);

private:
  Runtime *runtime_;
  TensorStore *store_;
  size_t deviceId_ = 0; ///< Device this Operations instance is bound to.

  // Graph state — Operations always records to graph_.
  std::unique_ptr<graph::Graph> graph_;

  // Cast deduplication cache — cleared whenever the graph is reset.
  struct CastCacheEntry {
    Tensor input;
    DataType targetDtype;
    Tensor output;
  };
  std::vector<CastCacheEntry> castCache_;

  std::vector<uint32_t> getShape(const Tensor &h) const;
  DataType getDtype(const Tensor &h) const;

  /// Replace the graph with a fresh one if the current one has been executed.
  void ensureFreshGraph();

  uint32_t toNodeId(const Tensor &t);

  Tensor recordOrEncode(std::unique_ptr<OpNode> node);

  /// Checks resolveInputDtypes() and inserts casts for mismatched inputs.
  /// Returns the (possibly cast) inputs. Uses castCache_ for deduplication.
  std::vector<Tensor> resolveAndCastInputs(const OpNode &node,
                                           const std::vector<Tensor> &inputs);

  /// Widen two inputs to matching dtype if they differ.
  /// Used by conv1d/conv2d before node construction.
  std::pair<Tensor, Tensor> widenToMatch(const Tensor &a, const Tensor &b);

  /// Create a standard MatMulOpNode with dtype resolution applied.
  /// Constructs the node once, resolves dtypes, and only reconstructs
  /// if inputs actually needed casting.
  std::unique_ptr<MatMulOpNode> createMatMulResolved(
      const Tensor &a, const Tensor &b, std::optional<uint32_t> spec);

  /// Create a quantized MatMulOpNode with dtype resolution applied.
  std::unique_ptr<MatMulOpNode>
  createMatMulResolved(const Tensor &a,
                       const Tensor &packedB,
                       const Tensor &scales,
                       std::optional<uint32_t> spec);
};

} // namespace cut
