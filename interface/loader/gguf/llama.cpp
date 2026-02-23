#include "llama.h"
#include "Operations.h"
#include "Runtime.h"
#include "model_report.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace gguf {

// ============================================================================
// Helpers
// ============================================================================

cut::ComputeHandle LlamaModel::uploadVector(const std::vector<float> &data) {
  std::vector<uint32_t> shape = {static_cast<uint32_t>(data.size())};
  return runtime_->createTensor(shape, cut::DataType::Float32, data.data());
}

cut::ComputeHandle
LlamaModel::uploadMatrix(const float *data, uint32_t rows, uint32_t cols) {
  std::vector<uint32_t> shape = {rows, cols};
  return runtime_->createTensor(shape, cut::DataType::Float32, data);
}

// ============================================================================
// Loading
// ============================================================================

void LlamaModel::load(const std::string &gguf_path, cut::Runtime &runtime) {
  runtime_ = &runtime;
  ops_ = &runtime.ops();

  std::cout << "Loading GGUF model: " << gguf_path << "\n";

  GGUFReader reader(gguf_path);
  const auto &meta = reader.metadata();
  std::string arch = meta.architecture();

  std::cout << "Architecture: " << arch << "\n";

  // Read config from metadata
  // Keys follow the pattern: {arch}.{param}
  std::string prefix = arch + ".";

  config_.dim = meta.get_as<uint32_t>(prefix + "embedding_length", 0);
  config_.n_layers = meta.get_as<uint32_t>(prefix + "block_count", 0);
  config_.n_heads = meta.get_as<uint32_t>(prefix + "attention.head_count", 0);
  config_.n_kv_heads = meta.get_as<uint32_t>(prefix + "attention.head_count_kv",
                                             config_.n_heads);
  config_.ffn_dim = meta.get_as<uint32_t>(prefix + "feed_forward_length", 0);
  config_.rope_freq_base =
      meta.get_as<float>(prefix + "rope.freq_base", 10000.0f);
  config_.norm_eps =
      meta.get_as<float>(prefix + "attention.layer_norm_rms_epsilon", 1e-5f);
  config_.max_seq_len = meta.get_as<uint32_t>(prefix + "context_length", 2048);

  if (config_.dim == 0 || config_.n_layers == 0 || config_.n_heads == 0) {
    throw std::runtime_error("Failed to read model config from GGUF metadata");
  }

  config_.head_dim = config_.dim / config_.n_heads;
  config_.kv_dim = config_.head_dim * config_.n_kv_heads;
  config_.n_rep = config_.n_heads / config_.n_kv_heads;

  // Infer ffn_dim from weight shape if not in metadata
  if (config_.ffn_dim == 0) {
    std::string gate_name = "blk.0.ffn_gate.weight";
    if (reader.has_tensor(gate_name)) {
      const auto &info = reader.get_tensor_info(gate_name);
      config_.ffn_dim = static_cast<uint32_t>(info.dimensions[0]);
    }
  }

  std::cout << "Config: dim=" << config_.dim << " n_layers=" << config_.n_layers
            << " n_heads=" << config_.n_heads
            << " n_kv_heads=" << config_.n_kv_heads
            << " ffn_dim=" << config_.ffn_dim
            << " head_dim=" << config_.head_dim << " vocab_size=";

  // Load token embeddings (keep on CPU for row lookup)
  // GGML dimensions: [cols=dim, rows=vocab_size]
  {
    auto embd = reader.read_tensor_f32("token_embd.weight");
    const auto &info = reader.get_tensor_info("token_embd.weight");
    config_.vocab_size = static_cast<uint32_t>(info.dimensions[1]);
    token_embd_data_ = std::move(embd);
  }
  std::cout << config_.vocab_size << "\n";

  // Load vocabulary from GGUF metadata
  if (meta.has("tokenizer.ggml.tokens")) {
    vocab_ = meta.get_as<std::vector<std::string>>("tokenizer.ggml.tokens");
  }

  // Load layers
  layers_.resize(config_.n_layers);
  for (uint32_t i = 0; i < config_.n_layers; ++i) {
    std::string blk = "blk." + std::to_string(i) + ".";
    auto &layer = layers_[i];

    std::cout << "  Loading layer " << i << "...\r" << std::flush;

    // Attention norm (1D weight)
    {
      auto data = reader.read_tensor_f32(blk + "attn_norm.weight");
      layer.attn_norm = uploadVector(data);
    }

    // Attention weights — GGUF/GGML dimensions are [cols, rows] (innermost
    // first). Data in memory is [rows, cols] row-major, i.e. [out_features,
    // in_features]. We need [in_features, out_features] for matmul(input[1,in],
    // W[in,out]), so we upload as [rows, cols] and transpose.
    {
      auto data = reader.read_tensor_f32(blk + "attn_q.weight");
      const auto &info = reader.get_tensor_info(blk + "attn_q.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      auto gpu = uploadMatrix(data.data(), rows, cols);
      layer.wq = ops_->transpose(gpu);
    }
    {
      auto data = reader.read_tensor_f32(blk + "attn_k.weight");
      const auto &info = reader.get_tensor_info(blk + "attn_k.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      auto gpu = uploadMatrix(data.data(), rows, cols);
      layer.wk = ops_->transpose(gpu);
    }
    {
      auto data = reader.read_tensor_f32(blk + "attn_v.weight");
      const auto &info = reader.get_tensor_info(blk + "attn_v.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      auto gpu = uploadMatrix(data.data(), rows, cols);
      layer.wv = ops_->transpose(gpu);
    }
    {
      auto data = reader.read_tensor_f32(blk + "attn_output.weight");
      const auto &info = reader.get_tensor_info(blk + "attn_output.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      auto gpu = uploadMatrix(data.data(), rows, cols);
      layer.wo = ops_->transpose(gpu);
    }

    // FFN norm
    {
      auto data = reader.read_tensor_f32(blk + "ffn_norm.weight");
      layer.ffn_norm = uploadVector(data);
    }

    // FFN weights (same transpose treatment)
    {
      auto data = reader.read_tensor_f32(blk + "ffn_gate.weight");
      const auto &info = reader.get_tensor_info(blk + "ffn_gate.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      auto gpu = uploadMatrix(data.data(), rows, cols);
      layer.w_gate = ops_->transpose(gpu);
    }
    {
      auto data = reader.read_tensor_f32(blk + "ffn_up.weight");
      const auto &info = reader.get_tensor_info(blk + "ffn_up.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      auto gpu = uploadMatrix(data.data(), rows, cols);
      layer.w_up = ops_->transpose(gpu);
    }
    {
      auto data = reader.read_tensor_f32(blk + "ffn_down.weight");
      const auto &info = reader.get_tensor_info(blk + "ffn_down.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      auto gpu = uploadMatrix(data.data(), rows, cols);
      layer.w_down = ops_->transpose(gpu);
    }
  }
  std::cout << "  Loaded all " << config_.n_layers << " layers.     \n";

  // Output norm
  {
    auto data = reader.read_tensor_f32("output_norm.weight");
    output_norm_ = uploadVector(data);
  }

  // Output weight (LM head)
  if (reader.has_tensor("output.weight")) {
    auto data = reader.read_tensor_f32("output.weight");
    const auto &info = reader.get_tensor_info("output.weight");
    uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
    uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
    auto gpu = uploadMatrix(data.data(), rows, cols);
    output_weight_ = ops_->transpose(gpu);
  } else {
    // Some models tie embeddings — use token_embd.weight transposed
    auto gpu =
        uploadMatrix(token_embd_data_.data(), config_.vocab_size, config_.dim);
    output_weight_ = ops_->transpose(gpu);
  }

  // Initialize KV caches
  kv_caches_.resize(config_.n_layers);

  // Precompute RoPE tables
  precomputeRoPE();

  runtime_->flush();
  std::cout << "Model loaded successfully. Buffers: " << runtime_->bufferCount()
            << "\n";

  // Generate HTML architecture report next to the model file.
  {
    std::string reportPath = gguf_path;
    auto dot = reportPath.rfind('.');
    if (dot != std::string::npos) {
      reportPath = reportPath.substr(0, dot);
    }
    reportPath += "_report.html";
    generateModelReport(reader, config_, reportPath);
  }
}

// ============================================================================
// RoPE precomputation
// ============================================================================

void LlamaModel::precomputeRoPE() {
  uint32_t half_dim = config_.head_dim / 2;
  rope_cos_.resize(config_.max_seq_len * half_dim);
  rope_sin_.resize(config_.max_seq_len * half_dim);

  for (uint32_t pos = 0; pos < config_.max_seq_len; ++pos) {
    for (uint32_t i = 0; i < half_dim; ++i) {
      float freq =
          1.0f / std::pow(config_.rope_freq_base,
                          static_cast<float>(2 * i) / config_.head_dim);
      float angle = static_cast<float>(pos) * freq;
      rope_cos_[pos * half_dim + i] = std::cos(angle);
      rope_sin_[pos * half_dim + i] = std::sin(angle);
    }
  }
}

// ============================================================================
// RMS Norm
// ============================================================================

cut::ComputeHandle LlamaModel::rmsNorm(const cut::ComputeHandle &x,
                                       const cut::ComputeHandle &weight) {
  // x is 1D [dim]
  // 1. Square elements
  auto x_sq = ops_->unaryOp(cut::UnarySquare, x);

  // 2. Sum and compute scale on CPU
  auto sumTensor = ops_->reduce(cut::ReduceSum, x_sq);
  float sum = 0.0f;
  runtime_->copyFromTensor(sumTensor, &sum, sizeof(float));
  float scale = 1.0f / std::sqrt(sum / static_cast<float>(config_.dim) +
                                 config_.norm_eps);

  // 3. Scale x
  auto normalized = ops_->vecScalarOp(cut::BinaryVecScalarMul, x, scale);

  // 4. Multiply by weight
  return ops_->binaryOp(cut::BinaryVecVecMul, normalized, weight);
}

// ============================================================================
// RoPE application
// ============================================================================

cut::ComputeHandle LlamaModel::applyRoPE(const cut::ComputeHandle &x,
                                         int pos,
                                         uint32_t n_heads_for_rope) {
  // x is 1D [n_heads_for_rope * head_dim]
  // Read back to CPU
  uint32_t total = n_heads_for_rope * config_.head_dim;
  std::vector<float> xdata(total);
  runtime_->copyFromTensor(x, xdata.data(), total * sizeof(float));

  uint32_t half_dim = config_.head_dim / 2;

  // Apply rotation to each head
  for (uint32_t h = 0; h < n_heads_for_rope; ++h) {
    float *head_ptr = xdata.data() + h * config_.head_dim;
    for (uint32_t i = 0; i < half_dim; ++i) {
      float cos_val = rope_cos_[pos * half_dim + i];
      float sin_val = rope_sin_[pos * half_dim + i];

      float x0 = head_ptr[i];
      float x1 = head_ptr[i + half_dim];

      head_ptr[i] = x0 * cos_val - x1 * sin_val;
      head_ptr[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
  }

  return uploadVector(xdata);
}

// ============================================================================
// Attention
// ============================================================================

cut::ComputeHandle LlamaModel::attention(const cut::ComputeHandle &q,
                                         const cut::ComputeHandle &k,
                                         const cut::ComputeHandle &v,
                                         int layer,
                                         int pos) {
  auto &cache = kv_caches_[layer];
  uint32_t head_dim = config_.head_dim;
  uint32_t n_heads = config_.n_heads;
  uint32_t kv_dim = config_.kv_dim;
  uint32_t n_rep = config_.n_rep;

  // Read K, V for current position back to CPU and append to cache
  std::vector<float> k_data(kv_dim);
  std::vector<float> v_data(kv_dim);
  runtime_->copyFromTensor(k, k_data.data(), kv_dim * sizeof(float));
  runtime_->copyFromTensor(v, v_data.data(), kv_dim * sizeof(float));

  cache.k_cache.insert(cache.k_cache.end(), k_data.begin(), k_data.end());
  cache.v_cache.insert(cache.v_cache.end(), v_data.begin(), v_data.end());
  cache.seq_len = static_cast<uint32_t>(pos) + 1;

  uint32_t seq_len = cache.seq_len;

  // Read Q back to CPU for per-head processing
  std::vector<float> q_data(n_heads * head_dim);
  runtime_->copyFromTensor(q, q_data.data(),
                           n_heads * head_dim * sizeof(float));

  // Output buffer for all heads concatenated
  std::vector<float> output(n_heads * head_dim, 0.0f);

  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Process each head
  for (uint32_t h = 0; h < n_heads; ++h) {
    uint32_t kv_h = h / n_rep; // KV head index for GQA

    // Q for this head: [head_dim]
    float *q_head = q_data.data() + h * head_dim;

    // Compute attention scores on CPU for simplicity
    std::vector<float> scores(seq_len);
    for (uint32_t t = 0; t < seq_len; ++t) {
      float dot = 0.0f;
      const float *k_t = cache.k_cache.data() + t * kv_dim + kv_h * head_dim;
      for (uint32_t d = 0; d < head_dim; ++d) {
        dot += q_head[d] * k_t[d];
      }
      scores[t] = dot * scale;
    }

    // Softmax on CPU
    float max_score = scores[0];
    for (uint32_t t = 1; t < seq_len; ++t) {
      if (scores[t] > max_score)
        max_score = scores[t];
    }
    float exp_sum = 0.0f;
    for (uint32_t t = 0; t < seq_len; ++t) {
      scores[t] = std::exp(scores[t] - max_score);
      exp_sum += scores[t];
    }
    for (uint32_t t = 0; t < seq_len; ++t) {
      scores[t] /= exp_sum;
    }

    // Weighted sum of V
    float *out_head = output.data() + h * head_dim;
    for (uint32_t t = 0; t < seq_len; ++t) {
      const float *v_t = cache.v_cache.data() + t * kv_dim + kv_h * head_dim;
      for (uint32_t d = 0; d < head_dim; ++d) {
        out_head[d] += scores[t] * v_t[d];
      }
    }
  }

  return uploadVector(output);
}

// ============================================================================
// SwiGLU FFN
// ============================================================================

cut::ComputeHandle LlamaModel::ffn(const cut::ComputeHandle &x, int layer) {
  auto &l = layers_[layer];

  // x is 1D [dim], reshape to [1, dim] for matmul
  auto x_2d = ops_->reshape(x, {1, static_cast<int32_t>(config_.dim)});

  // gate = x @ w_gate  [1, dim] x [dim, ffn_dim] = [1, ffn_dim]
  auto gate = ops_->matmul(x_2d, l.w_gate);

  // up = x @ w_up  [1, dim] x [dim, ffn_dim] = [1, ffn_dim]
  auto up = ops_->matmul(x_2d, l.w_up);

  // gate = silu(gate)
  gate = ops_->unaryOp(cut::UnarySilu, gate);

  // gate_up = gate * up (element-wise)
  auto gate_up = ops_->binaryOp(cut::BinaryVecVecMul, gate, up);

  // output = gate_up @ w_down  [1, ffn_dim] x [ffn_dim, dim] = [1, dim]
  auto out = ops_->matmul(gate_up, l.w_down);

  // Reshape back to 1D [dim]
  return ops_->reshape(out, {static_cast<int32_t>(config_.dim)});
}

// ============================================================================
// Forward pass
// ============================================================================

cut::ComputeHandle LlamaModel::forward(int token_id, int pos) {
  uint32_t dim = config_.dim;

  // 1. Token embedding lookup (CPU → GPU)
  const float *embd_row = token_embd_data_.data() + token_id * dim;
  std::vector<float> embd_vec(embd_row, embd_row + dim);
  auto hidden = uploadVector(embd_vec);

  // 2. Transformer layers
  for (uint32_t i = 0; i < config_.n_layers; ++i) {
    auto &l = layers_[i];

    // --- Attention block ---
    // RMS norm
    auto normed = rmsNorm(hidden, l.attn_norm);

    // Linear projections: reshape to [1, dim] for matmul
    auto normed_2d = ops_->reshape(normed, {1, static_cast<int32_t>(dim)});

    // Q = normed @ wq  [1, dim] x [dim, dim] = [1, dim]
    auto q = ops_->matmul(normed_2d, l.wq);
    q = ops_->reshape(
        q, {static_cast<int32_t>(config_.n_heads * config_.head_dim)});

    // K = normed @ wk  [1, dim] x [dim, kv_dim] = [1, kv_dim]
    auto k = ops_->matmul(normed_2d, l.wk);
    k = ops_->reshape(k, {static_cast<int32_t>(config_.kv_dim)});

    // V = normed @ wv  [1, dim] x [dim, kv_dim] = [1, kv_dim]
    auto v = ops_->matmul(normed_2d, l.wv);
    v = ops_->reshape(v, {static_cast<int32_t>(config_.kv_dim)});

    // Apply RoPE to Q and K
    q = applyRoPE(q, pos, config_.n_heads);
    k = applyRoPE(k, pos, config_.n_kv_heads);

    // Attention with KV cache
    auto attn_out = attention(q, k, v, i, pos);

    // Output projection: [1, dim] x [dim, dim] = [1, dim]
    auto attn_out_2d = ops_->reshape(attn_out, {1, static_cast<int32_t>(dim)});
    auto proj = ops_->matmul(attn_out_2d, l.wo);
    proj = ops_->reshape(proj, {static_cast<int32_t>(dim)});

    // Residual connection
    hidden = ops_->binaryOp(cut::BinaryVecVecAdd, hidden, proj);

    // --- FFN block ---
    auto normed_ffn = rmsNorm(hidden, l.ffn_norm);
    auto ffn_out = ffn(normed_ffn, i);

    // Residual connection
    hidden = ops_->binaryOp(cut::BinaryVecVecAdd, hidden, ffn_out);
  }

  // 3. Final RMS norm
  hidden = rmsNorm(hidden, output_norm_);

  // 4. LM head: [1, dim] x [dim, vocab] = [1, vocab]
  auto hidden_2d = ops_->reshape(hidden, {1, static_cast<int32_t>(dim)});
  auto logits = ops_->matmul(hidden_2d, output_weight_);

  return logits;
}

// ============================================================================
// Generation
// ============================================================================

std::vector<int> LlamaModel::generate(const std::vector<int> &prompt_tokens,
                                      int max_new_tokens) {
  resetCache();

  std::vector<int> tokens = prompt_tokens;
  int next_token = 0;

  // Process prompt tokens (prefill)
  for (size_t i = 0; i < prompt_tokens.size(); ++i) {
    auto logits = forward(prompt_tokens[i], static_cast<int>(i));

    // Only sample from last prompt token
    if (i == prompt_tokens.size() - 1) {
      // Argmax sampling
      auto argTensor = ops_->reduce(cut::ReduceArgmax, logits);
      float argF = 0.0f;
      runtime_->copyFromTensor(argTensor, &argF, sizeof(float));
      next_token = static_cast<int>(argF);
    }
  }

  tokens.push_back(next_token);
  std::cout << "Generated token: " << next_token << "\n";

  // Autoregressive generation
  for (int step = 0; step < max_new_tokens - 1; ++step) {
    int pos = static_cast<int>(prompt_tokens.size()) + step;
    auto logits = forward(next_token, pos);

    auto argTensor = ops_->reduce(cut::ReduceArgmax, logits);
    float argF = 0.0f;
    runtime_->copyFromTensor(argTensor, &argF, sizeof(float));
    next_token = static_cast<int>(argF);
    tokens.push_back(next_token);

    std::cout << "Generated token: " << next_token << "\n";

    // Stop on EOS (token 2 is common EOS for LLaMA)
    if (next_token == 2) {
      break;
    }
  }

  return tokens;
}

std::string LlamaModel::detokenize(const std::vector<int> &tokens) const {
  std::string result;
  for (int id : tokens) {
    if (id >= 0 && static_cast<size_t>(id) < vocab_.size()) {
      const auto &piece = vocab_[id];
      // SentencePiece uses \xe2\x96\x81 (U+2581) as the space marker
      for (size_t i = 0; i < piece.size();) {
        if (i + 2 < piece.size() && static_cast<uint8_t>(piece[i]) == 0xe2 &&
            static_cast<uint8_t>(piece[i + 1]) == 0x96 &&
            static_cast<uint8_t>(piece[i + 2]) == 0x81) {
          result += ' ';
          i += 3;
        } else {
          result += piece[i];
          i++;
        }
      }
    }
  }
  return result;
}

void LlamaModel::resetCache() {
  for (auto &cache : kv_caches_) {
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.seq_len = 0;
  }
}

} // namespace gguf
