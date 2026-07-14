#pragma once

#include "gguf_reader.hpp"

#include <ComputeCommon.h>
#include <ComputeHandle.h>
#include <graph/Graph.h>
#include <graph/GraphBuilder.h>
#include <graph/GraphExecutor.h>
#include <graph/GraphOptimizer.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cut {
class Runtime;
class Operations;
} // namespace cut

namespace gguf {

/// Model hyperparameters extracted from GGUF metadata.
struct LlamaConfig {
  uint32_t dim = 0;        // embedding_length (hidden dim)
  uint32_t n_layers = 0;   // block_count
  uint32_t n_heads = 0;    // attention.head_count
  uint32_t n_kv_heads = 0; // attention.head_count_kv (for GQA)
  uint32_t vocab_size = 0; // vocab_size (from token_embd shape)
  uint32_t ffn_dim = 0;    // feed_forward_length
  uint32_t head_dim = 0;   // dim / n_heads
  uint32_t kv_dim = 0;     // head_dim * n_kv_heads
  uint32_t n_rep = 0;      // n_heads / n_kv_heads (GQA repeat factor)
  float rope_freq_base = 10000.0f;
  float norm_eps = 1e-5f;
  uint32_t max_seq_len = 512; // default matches llama.cpp n_ctx
};

/// Weight handle that supports both plain and quantized storage.
struct WeightHandle {
  cut::ComputeHandle handle;  // plain weight (pre-transposed F16/F32)
  cut::ComputeHandle qValues; // Q8: Int8 [K,N] / Q4: packed nibbles [K,N/2]
  cut::ComputeHandle qScales; // F16 scales (transposed) [K/32, N]
  uint32_t qCols = 0;         // original K dimension (0 = not quantized)

  enum class QuantType { None, Q8_0, Q4_0 };
  QuantType quantType = QuantType::None;

  bool isQuantized() const { return quantType != QuantType::None; }
  bool isQ8() const { return quantType == QuantType::Q8_0; }
  bool isQ4() const { return quantType == QuantType::Q4_0; }
};

/// Per-layer weight handles stored on GPU.
struct LlamaLayer {
  // Attention norm
  cut::ComputeHandle attn_norm;

  // Attention weights
  WeightHandle wq;
  WeightHandle wk;
  WeightHandle wv;
  WeightHandle wo;

  // Fused QKV weight [dim, q_dim + 2*kv_dim] — set for non-quantized,
  // non-biased models. Enables single matmul + slice instead of 3 matmuls.
  WeightHandle wqkv;

  // Attention biases (optional, used by Qwen2 etc.)
  cut::ComputeHandle bq; // [dim] or empty
  cut::ComputeHandle bk; // [kv_dim] or empty
  cut::ComputeHandle bv; // [kv_dim] or empty

  // FFN norm
  cut::ComputeHandle ffn_norm;

  // FFN weights
  WeightHandle w_gate;
  WeightHandle w_up;
  WeightHandle w_down;

  // Fused gate+up weight [dim, 2*ffn_dim] — set for non-quantized F16 models.
  // Enables single matmul + slice instead of 2 separate matmuls.
  WeightHandle w_gate_up;
};

/// Per-layer KV cache stored on GPU.
struct KVCache {
  cut::ComputeHandle k_cache; // [max_seq_len, kv_dim] on GPU
  cut::ComputeHandle v_cache; // [max_seq_len, kv_dim] on GPU
  uint32_t seq_len = 0;       // Number of cached positions
};

/// A pre-built, optimized computation graph with handles to rebindable inputs.
struct GraphTemplate {
  std::unique_ptr<cut::graph::Graph> graph;
  cut::graph::Graph preOptGraph; // snapshot before optimization passes
  std::vector<uint32_t> dynamicInputIds;
  std::vector<cut::graph::PassStats> stats; // optimization statistics
};

/// Pre-built graph templates for one transformer layer.
struct LayerGraphs {
  GraphTemplate qkvProjection;
  GraphTemplate attnOutputResidual;
  GraphTemplate ffnResidual;
};

/// Result from autoregressive generation, including timing breakdown.
struct GenerationResult {
  std::vector<int> tokens; // all tokens (prompt + generated)
  double prefillMs = 0.0;  // time to process prompt tokens
  double generateMs = 0.0; // time for autoregressive generation only
  int promptTokens = 0;    // number of prompt tokens processed
  int generatedTokens = 0; // number of new tokens generated
};

/// LLaMA model that loads from GGUF and runs inference using CUT operators.
class LlamaModel {
public:
  LlamaModel() = default;
  ~LlamaModel() = default;

  // Non-copyable
  LlamaModel(const LlamaModel &) = delete;
  LlamaModel &operator=(const LlamaModel &) = delete;

  /// Load model from GGUF file onto GPU.
  /// @param gguf_path Path to the GGUF model file.
  /// @param runtime Initialized CUT runtime.
  /// @param n_ctx Context size (KV cache length). 0 = use default (512).
  ///              Capped to the model's context_length from GGUF metadata.
  void
  load(const std::string &gguf_path, cut::Runtime &runtime, uint32_t n_ctx = 0);

  /// Run a single forward pass for one token at the given position.
  /// @param token_id The input token ID.
  /// @param pos The sequence position (0-indexed).
  /// @return ComputeHandle to argmax result buffer.
  cut::ComputeHandle forward(int token_id, int pos);

  /// Prefill: process all prompt tokens in one command buffer submission.
  /// Populates KV cache for all positions. Returns argmax of last token.
  int prefill(const std::vector<int> &tokens);

  /// Batched prefill: processes all prompt tokens simultaneously using
  /// M=N matmul (GEMM) instead of N separate M=1 calls (GEMV).
  /// Gives ~10x prefill speedup on GPU.
  int prefillBatched(const std::vector<int> &tokens);

  /// Autoregressive generation from prompt tokens.
  /// @param prompt_tokens Input token IDs.
  /// @param max_new_tokens Maximum tokens to generate.
  /// @param repeat_penalty Repetition penalty (1.0 = disabled, >1.0 penalizes).
  /// @param repeat_last_n Lookback window for repetition penalty (0 = all).
  /// @return GenerationResult with tokens and timing breakdown.
  GenerationResult generate(const std::vector<int> &prompt_tokens,
                            int max_new_tokens,
                            float repeat_penalty = 1.05f,
                            int repeat_last_n = 64);

  /// Reset KV cache for a new generation.
  void resetCache();

  /// Encode text to token IDs using BPE (SentencePiece).
  std::vector<int> tokenize(const std::string &text) const;

  /// Decode token IDs to text using the GGUF vocabulary.
  std::string detokenize(const std::vector<int> &tokens) const;

  /// Access config.
  const LlamaConfig &config() const { return config_; }

  /// Look up a token string in the vocabulary, returns -1 if not found.
  int tokenId(const std::string &token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : -1;
  }

  int eosTokenId() const { return eos_token_id_; }

  /// Add extra stop token IDs (in addition to eos_token_id_).
  void addStopToken(int token_id) { stopTokenIds_.push_back(token_id); }

  /// Enable or disable per-dispatch GPU profiling.
  /// When enabled, each graph execution logs per-operation GPU timing.
  void setProfilingEnabled(bool enabled);

private:
  LlamaConfig config_;
  cut::Runtime *runtime_ = nullptr;

  // Weights
  cut::ComputeHandle embeddingTable_; // [vocab_size, dim] on GPU
  cut::ComputeHandle
      tokenIdBuffer_; // 1-element UInt32 for GPU embedding lookup
  std::vector<LlamaLayer> layers_;
  cut::ComputeHandle output_norm_;
  WeightHandle output_weight_; // LM head [dim, vocab_size]

  // KV cache (per layer)
  std::vector<KVCache> kv_caches_;

  // ---- Device placement (multi-GPU pipeline split) ----
  // Device id per layer (contiguous runs form pipeline segments). Parsed
  // from CUT_DEVICE_SPLIT ("n0,n1,..." = layer counts per runtime device);
  // default: all layers on device 0.
  std::vector<uint32_t> layerDevice_;
  // First layer index of each contiguous same-device segment.
  std::vector<uint32_t> segmentStart_;
  size_t firstDevice_ = 0; ///< Device of layer 0 (embedding lives here).
  size_t lastDevice_ = 0;  ///< Device of the last layer (logits/sampling).

  // ---- Per-device replicas, indexed by runtime device id ----
  // Only devices used by the placement are populated.
  std::vector<cut::ComputeHandle> ropeCosGpu_;
  std::vector<cut::ComputeHandle> ropeSinGpu_;
  std::vector<cut::ComputeHandle> runtimeParamsBuffers_; // {pos, seqLen} each
  std::vector<cut::ComputeHandle> hiddenBuffers_; // [dim] segment entry buffer
  std::vector<cut::ComputeHandle> attnOutBuffers_;

  // ---- Per-segment cached command buffers ----
  std::vector<cut::ComputeHandle> cachedDecodeCBs_;
  std::vector<cut::ComputeHandle> segmentDecodeOut_; // stable output per segment
  bool decodeCBCached_ = false;
  std::vector<cut::ComputeHandle> cachedPrefillCBs_;
  std::vector<cut::ComputeHandle> segmentPrefillOut_;
  bool prefillCBCached_ = false;

  // Per-device graph executors (indexed by device id).
  std::vector<std::unique_ptr<cut::graph::GraphExecutor>> executors_;

  // Vocabulary from GGUF metadata (token_id -> text)
  std::vector<std::string> vocab_;
  std::vector<float> scores_;
  std::unordered_map<std::string, int> token_to_id_;

  // Tokenizer type and BPE merge data
  std::string tokenizerModel_; // "llama" (SPM) or "gpt2" (BPE)
  std::vector<std::string> merges_;
  std::unordered_map<std::string, int> mergePriority_; // "a b" -> rank
  int bos_token_id_ = 1;
  bool addBosToken_ = true;
  int eos_token_id_ = 2;
  std::vector<int> stopTokenIds_;

  // Special tokens (e.g. <|im_start|>) sorted longest-first for matching
  std::vector<std::pair<std::string, int>> specialTokens_;

  // GPT-2 byte-level encoding tables (built when tokenizerModel_ == "gpt2")
  std::string byteToUnicode_[256];                         // byte → UTF-8 char
  std::unordered_map<std::string, uint8_t> unicodeToByte_; // reverse mapping

  // Pre-allocated buffers for command buffer reuse
  cut::ComputeHandle logitsOutput_;         // stable logits handle
  cut::ComputeHandle penaltyFactorsBuffer_; // [vocab_size] Float32
  cut::ComputeHandle argmaxResultBuffer_;   // [1] Float32 — argmax output

  // GPT-2 BPE tokenization (used when tokenizerModel_ == "gpt2")
  std::vector<int> tokenizeBPE(const std::string &text) const;
  void encodeBPESegment(const std::string &segment,
                        std::vector<int> &ids) const;
  void buildGPT2ByteEncoder();

  /// Prefill forward: embedding + layers only (no logits/penalty/argmax).
  /// Uses a separate cached CB from decode. Populates KV cache at pos.
  void forwardPrefill(int token_id, int pos);

  // Helper: upload 1D float vector to GPU
  cut::ComputeHandle uploadVector(const std::vector<float> &data, size_t deviceId);

  // Helper: upload 2D float data to GPU
  cut::ComputeHandle
  uploadMatrix(const float *data, uint32_t rows, uint32_t cols, size_t deviceId);

  // Helper: upload weight in native precision (F16 stays F16, etc.)
  cut::ComputeHandle uploadWeight(const GGUFReader &reader,
                                  const std::string &name,
                                  const std::vector<uint32_t> &shape,
                                  size_t deviceId);

  // Helper: concatenate multiple F16 tensors row-wise, upload, and transpose.
  // Used for fused QKV and fused gate+up weights.
  cut::ComputeHandle uploadFusedF16(const GGUFReader &reader,
                                    const std::vector<std::string> &names,
                                    size_t deviceId);

  // Helper: upload weight, using Q8 separated path if Q8_0
  WeightHandle uploadWeightMaybeQuantized(const GGUFReader &reader,
                                          const std::string &name,
                                          uint32_t rows,
                                          uint32_t cols,
                                          size_t deviceId);

  // Helper: perform matmul in graph using either plain or Q8 path
  cut::Tensor graphWeight(cut::graph::GraphBuilder &builder,
                          const WeightHandle &wh,
                          const cut::Tensor &activation);

  // Precompute RoPE tables
  void precomputeRoPE();

  // Graph-based execution
  std::vector<LayerGraphs> layerGraphs_;
  GraphTemplate logitsGraph_;

  void buildGraphTemplates();
  GraphTemplate buildQKVProjectionGraph(const LlamaLayer &layer, size_t deviceId);
  GraphTemplate buildAttnOutputResidualGraph(const LlamaLayer &layer, size_t deviceId);
  GraphTemplate buildFFNResidualGraph(const LlamaLayer &layer, size_t deviceId);
  GraphTemplate buildLogitsGraph();
  std::vector<cut::Tensor>
  executeGraph(GraphTemplate &tpl,
               const std::vector<cut::ComputeHandle> &dynamicHandles,
               size_t deviceId);

  /// Split fused QKV output into Q, K, V views (with barrier),
  /// or pass through separate Q/K/V from the graph.
  void splitQKV(const std::vector<cut::Tensor> &qkv,
                cut::ComputeHandle &q,
                cut::ComputeHandle &k,
                cut::ComputeHandle &v,
                size_t deviceId);

  /// Run one transformer layer: QKV projection → attention → FFN → residual.
  /// Returns the updated hidden state.
  cut::ComputeHandle runLayer(uint32_t layerIdx,
                              const cut::ComputeHandle &hidden);

  /// Operations for a given device id.
  cut::Operations &opsAt(size_t deviceId) const;

  /// Parse CUT_DEVICE_SPLIT and fill layerDevice_/segmentStart_/first/last.
  void computeLayerPlacement();

  /// Returns true when the given layer starts a new pipeline segment.
  bool isSegmentStart(uint32_t layerIdx) const;
};

} // namespace gguf
