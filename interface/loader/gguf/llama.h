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
  uint32_t max_seq_len = 2048;
};

/// Weight handle that supports both plain and quantized storage.
struct WeightHandle {
  cut::ComputeHandle handle;  // plain weight (pre-transposed F16/F32)
  cut::ComputeHandle qValues; // Int8 values (non-transposed) [N, K]
  cut::ComputeHandle qScales; // F16 scales (non-transposed) [N, K/32]
  uint32_t qCols = 0;         // original K dimension (0 = not quantized)
  bool isQuantized() const { return qCols > 0; }
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

  // FFN norm
  cut::ComputeHandle ffn_norm;

  // FFN weights
  WeightHandle w_gate;
  WeightHandle w_up;
  WeightHandle w_down;
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
};

/// Pre-built graph templates for one transformer layer.
struct LayerGraphs {
  GraphTemplate qkvProjection;
  GraphTemplate attnOutputResidual;
  GraphTemplate ffnResidual;
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
  void load(const std::string &gguf_path, cut::Runtime &runtime);

  /// Run a single forward pass for one token at the given position.
  /// @param token_id The input token ID.
  /// @param pos The sequence position (0-indexed).
  /// @return ComputeHandle to logits tensor [1, vocab_size].
  cut::ComputeHandle forward(int token_id, int pos);

  /// Autoregressive generation from prompt tokens.
  /// @param prompt_tokens Input token IDs.
  /// @param max_new_tokens Maximum tokens to generate.
  /// @param repeat_penalty Repetition penalty (1.0 = disabled, >1.0 penalizes).
  /// @param repeat_last_n Lookback window for repetition penalty (0 = all).
  /// @return All tokens (prompt + generated).
  std::vector<int> generate(const std::vector<int> &prompt_tokens,
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

private:
  LlamaConfig config_;
  cut::Runtime *runtime_ = nullptr;
  cut::Operations *ops_ = nullptr;

  // Weights
  std::vector<float> token_embd_data_; // [vocab_size, dim] kept on CPU
  std::vector<LlamaLayer> layers_;
  cut::ComputeHandle output_norm_;
  WeightHandle output_weight_; // LM head [dim, vocab_size]

  // KV cache (per layer)
  std::vector<KVCache> kv_caches_;

  // Precomputed RoPE cos/sin tables [max_seq_len * head_dim/2] on GPU
  cut::ComputeHandle rope_cos_gpu_;
  cut::ComputeHandle rope_sin_gpu_;

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

  // GPT-2 BPE tokenization (used when tokenizerModel_ == "gpt2")
  std::vector<int> tokenizeBPE(const std::string &text) const;
  void encodeBPESegment(const std::string &segment,
                        std::vector<int> &ids) const;
  void buildGPT2ByteEncoder();

  // Helper: upload 1D float vector to GPU
  cut::ComputeHandle uploadVector(const std::vector<float> &data);

  // Helper: upload 2D float data to GPU
  cut::ComputeHandle
  uploadMatrix(const float *data, uint32_t rows, uint32_t cols);

  // Helper: upload weight in native precision (F16 stays F16, etc.)
  cut::ComputeHandle uploadWeight(const GGUFReader &reader,
                                  const std::string &name,
                                  const std::vector<uint32_t> &shape);

  // Helper: upload weight, using Q8 separated path if Q8_0
  WeightHandle uploadWeightMaybeQuantized(const GGUFReader &reader,
                                          const std::string &name,
                                          uint32_t rows,
                                          uint32_t cols);

  // Helper: perform matmul in graph using either plain or Q8 path
  cut::Tensor graphWeight(cut::graph::GraphBuilder &builder,
                          const WeightHandle &wh,
                          const cut::Tensor &activation);

  // RMS normalization: returns normalized tensor
  cut::ComputeHandle rmsNorm(const cut::ComputeHandle &x,
                             const cut::ComputeHandle &weight);

  // Apply RoPE on GPU using precomputed cos/sin tables
  cut::ComputeHandle
  applyRoPE(const cut::ComputeHandle &x, int pos, uint32_t n_heads_for_rope);

  // Attention on GPU using KV cache
  cut::ComputeHandle attention(const cut::ComputeHandle &q,
                               const cut::ComputeHandle &k,
                               const cut::ComputeHandle &v,
                               int layer,
                               int pos);

  // Precompute RoPE tables
  void precomputeRoPE();

  // Graph-based execution
  std::vector<LayerGraphs> layerGraphs_;
  GraphTemplate logitsGraph_;
  std::unique_ptr<cut::graph::GraphExecutor> executor_;

  void buildGraphTemplates();
  GraphTemplate buildQKVProjectionGraph(const LlamaLayer &layer);
  GraphTemplate buildAttnOutputResidualGraph(const LlamaLayer &layer);
  GraphTemplate buildFFNResidualGraph(const LlamaLayer &layer);
  GraphTemplate buildLogitsGraph();
  std::vector<cut::Tensor>
  executeGraph(GraphTemplate &tpl,
               const std::vector<cut::ComputeHandle> &dynamicHandles);
};

} // namespace gguf
