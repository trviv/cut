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

/// Per-layer weight handles stored on GPU.
struct LlamaLayer {
  // Attention norm
  cut::ComputeHandle attn_norm;

  // Attention weights (stored transposed: [in, out] for matmul)
  cut::ComputeHandle wq;
  cut::ComputeHandle wk;
  cut::ComputeHandle wv;
  cut::ComputeHandle wo;

  // FFN norm
  cut::ComputeHandle ffn_norm;

  // FFN weights (stored transposed: [in, out] for matmul)
  cut::ComputeHandle w_gate;
  cut::ComputeHandle w_up;
  cut::ComputeHandle w_down;
};

/// Per-layer KV cache stored on CPU.
struct KVCache {
  // Each vector stores [position * kv_dim] floats (all KV heads concatenated)
  std::vector<float> k_cache;
  std::vector<float> v_cache;
  uint32_t seq_len = 0; // Number of cached positions
};

/// A pre-built, optimized computation graph with handles to rebindable inputs.
struct GraphTemplate {
  cut::graph::Graph graph;
  cut::graph::Graph preOptGraph; // snapshot before optimization passes
  std::vector<cut::graph::VirtualTensor> dynamicInputs;
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
  /// @return All tokens (prompt + generated).
  std::vector<int> generate(const std::vector<int> &prompt_tokens,
                            int max_new_tokens);

  /// Reset KV cache for a new generation.
  void resetCache();

  /// Decode token IDs to text using the GGUF vocabulary.
  std::string detokenize(const std::vector<int> &tokens) const;

  /// Access config.
  const LlamaConfig &config() const { return config_; }

private:
  LlamaConfig config_;
  cut::Runtime *runtime_ = nullptr;
  cut::Operations *ops_ = nullptr;

  // Weights
  std::vector<float> token_embd_data_; // [vocab_size, dim] kept on CPU
  std::vector<LlamaLayer> layers_;
  cut::ComputeHandle output_norm_;
  cut::ComputeHandle output_weight_; // LM head [dim, vocab_size]

  // KV cache (per layer)
  std::vector<KVCache> kv_caches_;

  // Precomputed RoPE cos/sin tables [max_seq_len, head_dim/2]
  std::vector<float> rope_cos_;
  std::vector<float> rope_sin_;

  // Vocabulary from GGUF metadata (token_id -> text)
  std::vector<std::string> vocab_;

  // Helper: upload 1D float vector to GPU
  cut::ComputeHandle uploadVector(const std::vector<float> &data);

  // Helper: upload 2D float data to GPU
  cut::ComputeHandle
  uploadMatrix(const float *data, uint32_t rows, uint32_t cols);

  // RMS normalization: returns normalized tensor
  cut::ComputeHandle rmsNorm(const cut::ComputeHandle &x,
                             const cut::ComputeHandle &weight);

  // Apply RoPE to Q or K vector on CPU, returns new GPU handle
  cut::ComputeHandle
  applyRoPE(const cut::ComputeHandle &x, int pos, uint32_t n_heads_for_rope);

  // Single-head or grouped attention computation
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
