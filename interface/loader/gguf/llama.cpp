#include "llama.h"
#include "OpNode.h"
#include "Operations.h"
#include "Runtime.h"
#include "model_report.h"

#include <algorithm>
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

  // Load vocabulary and tokenizer data from GGUF metadata
  if (meta.has("tokenizer.ggml.tokens")) {
    vocab_ = meta.get_as<std::vector<std::string>>("tokenizer.ggml.tokens");
    for (size_t i = 0; i < vocab_.size(); ++i) {
      token_to_id_[vocab_[i]] = static_cast<int>(i);
    }
  }
  if (meta.has("tokenizer.ggml.scores")) {
    scores_ = meta.get_as<std::vector<float>>("tokenizer.ggml.scores");
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

  // Build and optimize graph templates for forward pass
  buildGraphTemplates();

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

    // Collect optimized graph templates for the report (layer 0 as
    // representative).
    std::vector<NamedGraph> graphs;
    if (!layerGraphs_.empty()) {
      auto &lg = layerGraphs_[0];
      graphs.push_back({"QKV Projection", &lg.qkvProjection.preOptGraph,
                        &lg.qkvProjection.graph});
      graphs.push_back({"Attention Output + Residual",
                        &lg.attnOutputResidual.preOptGraph,
                        &lg.attnOutputResidual.graph});
      graphs.push_back({"FFN + Residual", &lg.ffnResidual.preOptGraph,
                        &lg.ffnResidual.graph});
    }
    graphs.push_back(
        {"Logits", &logitsGraph_.preOptGraph, &logitsGraph_.graph});
    generateModelReport(reader, config_, reportPath, graphs);
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
  auto normalized = ops_->binaryOp(cut::BinaryMul, x, scale);

  // 4. Multiply by weight
  return ops_->binaryOp(cut::BinaryMul, normalized, weight);
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
// Graph template builders
// ============================================================================

GraphTemplate LlamaModel::buildQKVProjectionGraph(const LlamaLayer &layer) {
  cut::graph::GraphBuilder builder(*runtime_);
  int32_t dim = static_cast<int32_t>(config_.dim);

  // Dynamic input: normed [dim] — use attn_norm as shape placeholder
  auto vNormed = builder.input(layer.attn_norm, /*isConstant=*/false);

  // Constant inputs: weight matrices
  auto vWq = builder.input(layer.wq, /*isConstant=*/true);
  auto vWk = builder.input(layer.wk, /*isConstant=*/true);
  auto vWv = builder.input(layer.wv, /*isConstant=*/true);

  auto normed_2d = builder.ops().reshape(vNormed, {1, dim});

  auto q = builder.ops().matmul(normed_2d, vWq);
  // Identity reshape (optimizer: IdentityReshapePass eliminates)
  auto q_id = builder.ops().reshape(q, {1, dim});
  // Reshape chain with above (optimizer: ReshapeChainPass collapses)
  auto q_flat = builder.ops().reshape(
      q_id, {static_cast<int32_t>(config_.n_heads * config_.head_dim)});

  auto k = builder.ops().matmul(normed_2d, vWk);
  auto k_id =
      builder.ops().reshape(k, {1, static_cast<int32_t>(config_.kv_dim)});
  auto k_flat =
      builder.ops().reshape(k_id, {static_cast<int32_t>(config_.kv_dim)});

  auto v = builder.ops().matmul(normed_2d, vWv);
  auto v_id =
      builder.ops().reshape(v, {1, static_cast<int32_t>(config_.kv_dim)});
  auto v_flat =
      builder.ops().reshape(v_id, {static_cast<int32_t>(config_.kv_dim)});

  // Dead code: result never used (optimizer: DeadCodePass removes)
  builder.ops().transpose(q);

  builder.markOutput(q_flat);
  builder.markOutput(k_flat);
  builder.markOutput(v_flat);

  auto graph = builder.build();

  GraphTemplate tpl;
  tpl.dynamicInputIds = {graph.nodeId(vNormed)};
  tpl.preOptGraph = graph.clone();
  auto optimizer = cut::graph::GraphOptimizer::createDefault();
  optimizer.optimize(graph);
  tpl.graph = std::move(graph);
  return tpl;
}

GraphTemplate
LlamaModel::buildAttnOutputResidualGraph(const LlamaLayer &layer) {
  cut::graph::GraphBuilder builder(*runtime_);
  int32_t dim = static_cast<int32_t>(config_.dim);

  // Dynamic inputs — each must use a DIFFERENT placeholder tensor so that
  // Operations can distinguish them during graph construction.
  // Using the same handle for both would cause the residual add to read from
  // the wrong input (attn_out instead of hidden).
  auto vAttnOut = builder.input(layer.attn_norm, /*isConstant=*/false);
  auto vWo = builder.input(layer.wo, /*isConstant=*/true);
  auto vHidden = builder.input(layer.ffn_norm, /*isConstant=*/false);

  auto attn_2d = builder.ops().reshape(vAttnOut, {1, dim});
  auto proj = builder.ops().matmul(attn_2d, vWo);
  // Identity reshape (optimizer: IdentityReshapePass eliminates)
  auto proj_id = builder.ops().reshape(proj, {1, dim});
  // Reshape chain (optimizer: ReshapeChainPass collapses)
  auto proj_1d = builder.ops().reshape(proj_id, {dim});
  auto result = builder.ops().binaryOp(cut::BinaryAdd, vHidden, proj_1d);

  // Dead code: unused transpose (optimizer: DeadCodePass removes)
  builder.ops().transpose(proj);

  builder.markOutput(result);

  auto graph = builder.build();

  GraphTemplate tpl;
  tpl.dynamicInputIds = {graph.nodeId(vAttnOut), graph.nodeId(vHidden)};
  tpl.preOptGraph = graph.clone();
  auto optimizer = cut::graph::GraphOptimizer::createDefault();
  optimizer.optimize(graph);
  tpl.graph = std::move(graph);
  return tpl;
}

GraphTemplate LlamaModel::buildFFNResidualGraph(const LlamaLayer &layer) {
  cut::graph::GraphBuilder builder(*runtime_);
  int32_t dim = static_cast<int32_t>(config_.dim);

  // Dynamic inputs — each must use a DIFFERENT placeholder tensor so that
  // Operations can distinguish them during graph construction.
  auto vNormed = builder.input(layer.ffn_norm, /*isConstant=*/false);
  auto vWGate = builder.input(layer.w_gate, /*isConstant=*/true);
  auto vWUp = builder.input(layer.w_up, /*isConstant=*/true);
  auto vWDown = builder.input(layer.w_down, /*isConstant=*/true);
  auto vHidden = builder.input(layer.attn_norm, /*isConstant=*/false);

  auto x_2d = builder.ops().reshape(vNormed, {1, dim});

  int32_t ffn = static_cast<int32_t>(config_.ffn_dim);

  auto gate = builder.ops().matmul(x_2d, vWGate);
  // Identity reshape (optimizer: IdentityReshapePass eliminates)
  auto gate_id = builder.ops().reshape(gate, {1, ffn});
  auto up = builder.ops().matmul(x_2d, vWUp);
  // Identity reshape (optimizer: IdentityReshapePass eliminates)
  auto up_id = builder.ops().reshape(up, {1, ffn});

  auto gate_silu = builder.ops().unaryOp(cut::UnarySilu, gate_id);
  auto gate_up = builder.ops().binaryOp(cut::BinaryMul, gate_silu, up_id);

  auto out = builder.ops().matmul(gate_up, vWDown);
  // Identity reshape + reshape chain (optimizer eliminates both)
  auto out_id = builder.ops().reshape(out, {1, dim});
  auto out_1d = builder.ops().reshape(out_id, {dim});

  // Dead code: unused transpose (optimizer: DeadCodePass removes)
  builder.ops().transpose(gate);

  auto result = builder.ops().binaryOp(cut::BinaryAdd, vHidden, out_1d);

  builder.markOutput(result);

  auto graph = builder.build();

  GraphTemplate tpl;
  tpl.dynamicInputIds = {graph.nodeId(vNormed), graph.nodeId(vHidden)};
  tpl.preOptGraph = graph.clone();
  auto optimizer = cut::graph::GraphOptimizer::createDefault();
  optimizer.optimize(graph);
  tpl.graph = std::move(graph);
  return tpl;
}

GraphTemplate LlamaModel::buildLogitsGraph() {
  cut::graph::GraphBuilder builder(*runtime_);
  int32_t dim = static_cast<int32_t>(config_.dim);

  // Dynamic input — use output_norm_ as shape placeholder for {dim}
  auto vHidden = builder.input(output_norm_, /*isConstant=*/false);
  auto vOutWeight = builder.input(output_weight_, /*isConstant=*/true);

  auto hidden_2d = builder.ops().reshape(vHidden, {1, dim});
  auto logits_raw = builder.ops().matmul(hidden_2d, vOutWeight);

  // Dead code: identity reshape (optimizer: IdentityReshapePass + DeadCodePass)
  builder.ops().reshape(logits_raw,
                        {1, static_cast<int32_t>(config_.vocab_size)});

  // Dead code: unused transpose (optimizer: DeadCodePass removes)
  builder.ops().transpose(logits_raw);

  builder.markOutput(logits_raw);

  auto graph = builder.build();

  GraphTemplate tpl;
  tpl.dynamicInputIds = {graph.nodeId(vHidden)};
  tpl.preOptGraph = graph.clone();
  auto optimizer = cut::graph::GraphOptimizer::createDefault();
  optimizer.optimize(graph);
  tpl.graph = std::move(graph);
  return tpl;
}

void LlamaModel::buildGraphTemplates() {
  executor_ = std::make_unique<cut::graph::GraphExecutor>(*ops_, *runtime_);

  layerGraphs_.resize(config_.n_layers);
  for (uint32_t i = 0; i < config_.n_layers; ++i) {
    layerGraphs_[i].qkvProjection = buildQKVProjectionGraph(layers_[i]);
    layerGraphs_[i].attnOutputResidual =
        buildAttnOutputResidualGraph(layers_[i]);
    layerGraphs_[i].ffnResidual = buildFFNResidualGraph(layers_[i]);
  }
  logitsGraph_ = buildLogitsGraph();

  std::cout << "Built and optimized " << (config_.n_layers * 3 + 1)
            << " graph templates.\n";
}

std::vector<cut::Tensor> LlamaModel::executeGraph(
    GraphTemplate &tpl, const std::vector<cut::ComputeHandle> &dynamicHandles) {
  for (size_t i = 0; i < tpl.dynamicInputIds.size(); ++i) {
    auto &node = tpl.graph.node(tpl.dynamicInputIds[i]);
    static_cast<cut::InputOpNode &>(node).setGpuHandle(dynamicHandles[i]);
  }
  return executor_->execute(tpl.graph);
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

  // Debug: trace where hidden state converges
  static int fwdDbgCount = 0;
  if (fwdDbgCount < 3) {
    float dbg[4];
    runtime_->copyFromTensor(hidden, dbg, 4 * sizeof(float));
    std::cerr << "  [fwd " << fwdDbgCount << "] tok=" << token_id
              << " pos=" << pos << " emb=[" << dbg[0] << " " << dbg[1] << " "
              << dbg[2] << " " << dbg[3] << "]\n";
  }

  // 2. Transformer layers
  for (uint32_t i = 0; i < config_.n_layers; ++i) {
    auto &l = layers_[i];
    auto &lg = layerGraphs_[i];

    // --- Attention block ---
    // RMS norm (CPU sync point — stays as direct ops)
    auto normed = rmsNorm(hidden, l.attn_norm);

    // QKV projection (graph template)
    auto qkv = executeGraph(lg.qkvProjection, {normed});
    auto q = qkv[0]; // [n_heads * head_dim]
    auto k = qkv[1]; // [kv_dim]
    auto v = qkv[2]; // [kv_dim]

    // Apply RoPE (CPU sync point — stays as direct ops)
    q = applyRoPE(q, pos, config_.n_heads);
    k = applyRoPE(k, pos, config_.n_kv_heads);

    // Attention with KV cache (CPU sync point — stays as direct ops)
    auto attn_out = attention(q, k, v, i, pos);

    // Output projection + residual (graph template)
    auto attn_result = executeGraph(lg.attnOutputResidual, {attn_out, hidden});
    hidden = attn_result[0];

    // --- FFN block ---
    // RMS norm (CPU sync point — stays as direct ops)
    auto normed_ffn = rmsNorm(hidden, l.ffn_norm);

    // FFN + residual (graph template)
    auto ffn_result = executeGraph(lg.ffnResidual, {normed_ffn, hidden});
    hidden = ffn_result[0];
  }

  // Debug: hidden state after all layers
  if (fwdDbgCount < 3) {
    float dbg[4];
    runtime_->copyFromTensor(hidden, dbg, 4 * sizeof(float));
    std::cerr << "  [fwd " << fwdDbgCount << "] after 30 layers=[" << dbg[0]
              << " " << dbg[1] << " " << dbg[2] << " " << dbg[3] << "]\n";
    ++fwdDbgCount;
  }

  // 3. Final RMS norm (CPU sync point — stays as direct ops)
  hidden = rmsNorm(hidden, output_norm_);

  // 4. LM head logits (graph template)
  auto logit_result = executeGraph(logitsGraph_, {hidden});
  return logit_result[0];
}

// ============================================================================
// Generation
// ============================================================================

std::vector<int> LlamaModel::generate(const std::vector<int> &prompt_tokens,
                                      int max_new_tokens,
                                      float repeat_penalty,
                                      int repeat_last_n) {
  resetCache();

  std::vector<int> tokens = prompt_tokens;
  int next_token = 0;
  uint32_t vocabSize = config_.vocab_size;

  // Sample with repetition penalty applied on CPU.
  // Algorithm (matching llama.cpp):
  //   For each token in the last repeat_last_n tokens:
  //     if logit > 0: logit /= repeat_penalty
  //     if logit < 0: logit *= repeat_penalty
  auto sample = [&](const cut::ComputeHandle &logits) -> int {
    std::vector<float> allLogits(vocabSize);
    runtime_->copyFromTensor(logits, allLogits.data(),
                             vocabSize * sizeof(float));

    // Apply repetition penalty
    if (repeat_penalty != 1.0f) {
      size_t start = 0;
      if (repeat_last_n > 0 &&
          tokens.size() > static_cast<size_t>(repeat_last_n)) {
        start = tokens.size() - static_cast<size_t>(repeat_last_n);
      }
      for (size_t i = start; i < tokens.size(); ++i) {
        int tid = tokens[i];
        if (tid >= 0 && static_cast<uint32_t>(tid) < vocabSize) {
          if (allLogits[tid] > 0.0f) {
            allLogits[tid] /= repeat_penalty;
          } else {
            allLogits[tid] *= repeat_penalty;
          }
        }
      }
    }

    // Argmax
    int best = 0;
    for (uint32_t j = 1; j < vocabSize; ++j) {
      if (allLogits[j] > allLogits[best]) {
        best = static_cast<int>(j);
      }
    }
    return best;
  };

  // Process prompt tokens (prefill)
  for (size_t i = 0; i < prompt_tokens.size(); ++i) {
    auto logits = forward(prompt_tokens[i], static_cast<int>(i));

    // Only sample from last prompt token
    if (i == prompt_tokens.size() - 1) {
      next_token = sample(logits);
    }
  }

  tokens.push_back(next_token);
  std::cout << "Generated token: " << next_token << "\n";

  // Stop on EOS from first sampled token
  if (next_token == 2) {
    return tokens;
  }

  // Autoregressive generation
  for (int step = 0; step < max_new_tokens - 1; ++step) {
    int pos = static_cast<int>(prompt_tokens.size()) + step;
    auto logits = forward(next_token, pos);

    next_token = sample(logits);
    tokens.push_back(next_token);

    std::cout << "Generated token: " << next_token << "\n";

    // Stop on EOS (token 2 is common EOS for LLaMA)
    if (next_token == 2) {
      break;
    }
  }

  return tokens;
}

// ============================================================================
// Tokenization (SentencePiece BPE)
// ============================================================================

// Advance past one UTF-8 character, returning byte count (1-4).
static int utf8_len(uint8_t lead) {
  if ((lead & 0x80) == 0)
    return 1;
  if ((lead & 0xE0) == 0xC0)
    return 2;
  if ((lead & 0xF0) == 0xE0)
    return 3;
  return 4;
}

std::vector<int> LlamaModel::tokenize(const std::string &text) const {
  if (vocab_.empty() || scores_.empty()) {
    throw std::runtime_error("Tokenizer data not loaded from GGUF");
  }

  // SentencePiece convention: replace spaces with U+2581 (▁) and prepend one
  // to mark word boundaries.
  static const std::string SP_SPACE = "\xe2\x96\x81"; // ▁
  std::string normalized;
  normalized += SP_SPACE;
  for (char c : text) {
    if (c == ' ') {
      normalized += SP_SPACE;
    } else {
      normalized += c;
    }
  }

  // Split into individual UTF-8 characters, falling back to byte tokens
  // (<0xNN>) for characters not in the vocabulary.
  std::vector<std::string> symbols;
  for (size_t i = 0; i < normalized.size();) {
    int len = utf8_len(static_cast<uint8_t>(normalized[i]));
    if (i + len > normalized.size())
      len = 1; // truncated — treat as single byte
    std::string ch = normalized.substr(i, len);
    if (token_to_id_.count(ch)) {
      symbols.push_back(ch);
    } else {
      // Byte fallback: emit <0xNN> tokens for each byte
      for (int b = 0; b < len; ++b) {
        char hex[8];
        std::snprintf(hex, sizeof(hex), "<0x%02X>",
                      static_cast<uint8_t>(normalized[i + b]));
        symbols.push_back(std::string(hex));
      }
    }
    i += len;
  }

  // BPE merge loop: repeatedly merge the adjacent pair whose merged result
  // has the highest score in the vocabulary, until no more merges exist.
  while (symbols.size() >= 2) {
    float best_score = -1e30f;
    size_t best_idx = SIZE_MAX;

    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      std::string merged = symbols[i] + symbols[i + 1];
      auto it = token_to_id_.find(merged);
      if (it != token_to_id_.end()) {
        float score = scores_[it->second];
        if (score > best_score) {
          best_score = score;
          best_idx = i;
        }
      }
    }

    if (best_idx == SIZE_MAX)
      break; // no more merges possible

    // Apply the merge
    symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
    symbols.erase(symbols.begin() + best_idx + 1);
  }

  // Convert symbols to token IDs, prepending BOS (token 1)
  std::vector<int> ids;
  ids.push_back(1); // BOS
  for (const auto &sym : symbols) {
    auto it = token_to_id_.find(sym);
    if (it != token_to_id_.end()) {
      ids.push_back(it->second);
    }
    // skip unknown symbols (shouldn't happen with byte fallback)
  }

  return ids;
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
