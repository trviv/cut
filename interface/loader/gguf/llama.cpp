#include "llama.h"
#include "OpNode.h"
#include "Operations.h"
#include "Runtime.h"
#include "impl/dequant/DequantOp.h"
#include "model_report.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <iostream>
#include <map>
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

cut::ComputeHandle
LlamaModel::uploadFusedF16(const GGUFReader &reader,
                           const std::vector<std::string> &names) {
  // Read multiple F16 tensors, concatenate rows, upload once, transpose once.
  // All tensors must share the same innermost dimension K (= cols in GGUF).
  uint32_t K =
      static_cast<uint32_t>(reader.get_tensor_info(names[0]).dimensions[0]);
  uint32_t totalRows = 0;
  for (const auto &name : names) {
    totalRows +=
        static_cast<uint32_t>(reader.get_tensor_info(name).dimensions[1]);
  }

  std::vector<uint8_t> combined(totalRows * K * 2);
  size_t off = 0;
  for (const auto &name : names) {
    auto raw = reader.read_tensor_raw(name);
    memcpy(combined.data() + off, raw.data(), raw.size());
    off += raw.size();
  }

  auto gpu = runtime_->createTensor({totalRows, K}, cut::DataType::Float16,
                                    combined.data());
  return ops_->transpose(gpu);
}

cut::ComputeHandle
LlamaModel::uploadWeight(const GGUFReader &reader,
                         const std::string &name,
                         const std::vector<uint32_t> &shape) {
  const auto &info = reader.get_tensor_info(name);

  if (info.type == GGMLType::F16) {
    // Upload Float16 weights directly — no conversion needed.
    auto raw = reader.read_tensor_raw(name);
    return runtime_->createTensor(shape, cut::DataType::Float16, raw.data());
  }

  // GPU dequantization for 2D weight matrices in K-quant or BF16 formats.
  // Upload raw bytes to GPU, then dequantize via compute shader.
  if (shape.size() == 2) {
    cut::DequantFormat fmt;
    bool useGpuDequant = true;
    switch (info.type) {
    case GGMLType::BF16:
      fmt = cut::DequantFormat::BF16;
      break;
    case GGMLType::Q4_K:
      fmt = cut::DequantFormat::Q4_K;
      break;
    case GGMLType::Q5_K:
      fmt = cut::DequantFormat::Q5_K;
      break;
    case GGMLType::Q6_K:
      fmt = cut::DequantFormat::Q6_K;
      break;
    default:
      useGpuDequant = false;
      break;
    }
    if (useGpuDequant) {
      auto raw = reader.read_tensor_raw(name);
      auto rawTensor = runtime_->createTensor(
          {static_cast<uint32_t>(raw.size())}, cut::DataType::Int8, raw.data());
      return ops_->dequantize(rawTensor, static_cast<uint32_t>(fmt), shape[0],
                              shape[1]);
    }
  }

  // Fallback: CPU dequantize to Float32 (F32 pass-through, small tensors).
  auto data = reader.read_tensor_f32(name);
  return runtime_->createTensor(shape, cut::DataType::Float32, data.data());
}

WeightHandle LlamaModel::uploadWeightMaybeQuantized(const GGUFReader &reader,
                                                    const std::string &name,
                                                    uint32_t rows,
                                                    uint32_t cols) {
  const auto &info = reader.get_tensor_info(name);
  WeightHandle wh;

  if (info.type == GGMLType::Q4_0) {
    // Upload Q4_0 weights and GPU-transpose packed nibbles + scales.
    // GGUF layout: [rows=N, cols=K]. Q4 has 2 nibbles per byte.
    auto q4 = reader.read_tensor_q4_separated(name);
    uint32_t N = rows, K = cols;
    uint32_t blocksK = K / 32;

    // Upload packed nibbles [N, K/2] directly to GPU
    auto gpuPacked = runtime_->createTensor({N, K / 2}, cut::DataType::Int8,
                                            q4.packedValues.data());

    // GPU nibble transpose: [N, K/2] -> [K, N/2]
    // Combines unpack (GGML block layout) + transpose + repack in one dispatch
    auto tPacked = ops_->transposeQ4(gpuPacked, N, K);

    // GPU transpose scales [N, K/32] -> [K/32, N]
    auto gpuScales = runtime_->createTensor(
        {N, blocksK}, cut::DataType::Float16, q4.scales.data());
    auto tScales = ops_->transpose(gpuScales);

    wh.qValues = tPacked;
    wh.qScales = tScales;
    wh.qCols = cols;
    wh.quantType = WeightHandle::QuantType::Q4_0;
    return wh;
  }

  if (info.type == GGMLType::Q8_0) {
    // Upload Q8_0 weights in GGUF-native [N, K] layout, then GPU-transpose
    // to [K, N] to match the matmul layout convention.
    auto q8 = reader.read_tensor_q8_separated(name);
    uint32_t N = rows, K = cols;
    uint32_t blocksK = K / 32;

    auto gpuValues =
        runtime_->createTensor({N, K}, cut::DataType::Int8, q8.values.data());
    auto gpuScales = runtime_->createTensor(
        {N, blocksK}, cut::DataType::Float16, q8.scales.data());
    wh.qValues = ops_->transpose(gpuValues);
    wh.qScales = ops_->transpose(gpuScales);
    wh.qCols = cols;
    wh.quantType = WeightHandle::QuantType::Q8_0;
    return wh;
  }

  // Non-quantized: upload + transpose as before
  auto gpu = uploadWeight(reader, name, {rows, cols});
  wh.handle = ops_->transpose(gpu);
  return wh;
}

cut::Tensor LlamaModel::graphWeight(cut::graph::GraphBuilder &builder,
                                    const WeightHandle &wh,
                                    const cut::Tensor &activation) {
  if (wh.isQuantized()) {
    auto vValues = builder.input(wh.qValues, /*isConstant=*/true);
    auto vScales = builder.input(wh.qScales, /*isConstant=*/true);
    return builder.ops().matmul(activation, vValues, vScales);
  }
  auto vW = builder.input(wh.handle, /*isConstant=*/true);
  return builder.ops().matmul(activation, vW);
}

// ============================================================================
// Loading
// ============================================================================

void LlamaModel::load(const std::string &gguf_path,
                      cut::Runtime &runtime,
                      uint32_t n_ctx) {
  runtime_ = &runtime;
  ops_ = &runtime.ops();

  auto loadStart = std::chrono::high_resolution_clock::now();
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

  // Context size: use n_ctx if provided, otherwise keep struct default (512).
  // Cap to model's context_length so RoPE tables and KV cache stay in bounds.
  uint32_t model_ctx = meta.get_as<uint32_t>(prefix + "context_length", 2048);
  if (n_ctx > 0) {
    config_.max_seq_len = n_ctx;
  }
  if (config_.max_seq_len > model_ctx) {
    std::cout << "WARNING: n_ctx=" << config_.max_seq_len
              << " exceeds model context_length=" << model_ctx << ", capping\n";
    config_.max_seq_len = model_ctx;
  }

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
            << " head_dim=" << config_.head_dim
            << " ctx=" << config_.max_seq_len << " vocab_size=";

  // Load token embeddings to GPU for GPU-side embedding lookup.
  // GGML dimensions: [cols=dim, rows=vocab_size]
  {
    auto embd = reader.read_tensor_f32("token_embd.weight");
    const auto &info = reader.get_tensor_info("token_embd.weight");
    config_.vocab_size = static_cast<uint32_t>(info.dimensions[1]);
    embeddingTable_ =
        uploadMatrix(embd.data(), config_.vocab_size, config_.dim);
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

  // Tokenizer model type: "llama" (SentencePiece) or "gpt2" (BPE)
  tokenizerModel_ = meta.get_as<std::string>("tokenizer.ggml.model", "llama");
  bos_token_id_ =
      static_cast<int>(meta.get_as<uint32_t>("tokenizer.ggml.bos_token_id", 1));
  eos_token_id_ =
      static_cast<int>(meta.get_as<uint32_t>("tokenizer.ggml.eos_token_id", 2));
  addBosToken_ = meta.get_as<bool>("tokenizer.ggml.add_bos_token", true);

  if (meta.has("tokenizer.ggml.merges")) {
    merges_ = meta.get_as<std::vector<std::string>>("tokenizer.ggml.merges");
    for (size_t i = 0; i < merges_.size(); ++i) {
      mergePriority_[merges_[i]] = static_cast<int>(i);
    }
  }

  if (tokenizerModel_ == "gpt2") {
    buildGPT2ByteEncoder();
  }

  // Collect special tokens (e.g. <|im_start|>, <s>, </s>) for literal matching
  for (size_t i = 0; i < vocab_.size(); ++i) {
    const auto &tok = vocab_[i];
    if (tok.size() >= 3 && tok.front() == '<' && tok.back() == '>') {
      specialTokens_.push_back({tok, static_cast<int>(i)});
    }
  }
  std::sort(specialTokens_.begin(), specialTokens_.end(),
            [](const auto &a, const auto &b) {
              return a.first.size() > b.first.size();
            });

  std::cout << "Tokenizer: model=" << tokenizerModel_
            << " vocab=" << vocab_.size() << " merges=" << merges_.size()
            << " bos=" << bos_token_id_ << " eos=" << eos_token_id_ << "\n";

  // Load layers
  auto layersStart = std::chrono::high_resolution_clock::now();
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

    // Attention biases (optional, e.g. Qwen2)
    bool hasBias = reader.has_tensor(blk + "attn_q.bias");
    if (hasBias) {
      layer.bq = uploadVector(reader.read_tensor_f32(blk + "attn_q.bias"));
      layer.bk = uploadVector(reader.read_tensor_f32(blk + "attn_k.bias"));
      layer.bv = uploadVector(reader.read_tensor_f32(blk + "attn_v.bias"));
    }

    // Attention weights — GGUF/GGML dimensions are [cols, rows] (innermost
    // first). For F16 without bias, build fused QKV directly (single upload +
    // single transpose) instead of uploading Q/K/V separately. This halves
    // attention weight GPU memory vs the old approach.
    {
      const auto &qi = reader.get_tensor_info(blk + "attn_q.weight");
      bool canFuse = !hasBias && qi.type == GGMLType::F16;

      if (canFuse) {
        layer.wqkv.handle = uploadFusedF16(reader, {
                                                       blk + "attn_q.weight",
                                                       blk + "attn_k.weight",
                                                       blk + "attn_v.weight",
                                                   });
      } else {
        // Separate Q/K/V (quantized or biased models)
        uint32_t qCols = static_cast<uint32_t>(qi.dimensions[0]);
        uint32_t qRows = static_cast<uint32_t>(qi.dimensions[1]);
        layer.wq = uploadWeightMaybeQuantized(reader, blk + "attn_q.weight",
                                              qRows, qCols);
        {
          const auto &info = reader.get_tensor_info(blk + "attn_k.weight");
          layer.wk = uploadWeightMaybeQuantized(
              reader, blk + "attn_k.weight",
              static_cast<uint32_t>(info.dimensions[1]),
              static_cast<uint32_t>(info.dimensions[0]));
        }
        {
          const auto &info = reader.get_tensor_info(blk + "attn_v.weight");
          layer.wv = uploadWeightMaybeQuantized(
              reader, blk + "attn_v.weight",
              static_cast<uint32_t>(info.dimensions[1]),
              static_cast<uint32_t>(info.dimensions[0]));
        }
      }
    }

    {
      const auto &info = reader.get_tensor_info(blk + "attn_output.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      layer.wo = uploadWeightMaybeQuantized(reader, blk + "attn_output.weight",
                                            rows, cols);
    }

    // FFN norm
    {
      auto data = reader.read_tensor_f32(blk + "ffn_norm.weight");
      layer.ffn_norm = uploadVector(data);
    }

    // FFN weights — for F16, build fused gate+up directly (single upload).
    {
      const auto &gi = reader.get_tensor_info(blk + "ffn_gate.weight");
      bool canFuseFFN = gi.type == GGMLType::F16;

      if (canFuseFFN) {
        layer.w_gate_up.handle =
            uploadFusedF16(reader, {
                                       blk + "ffn_gate.weight",
                                       blk + "ffn_up.weight",
                                   });
      } else {
        // Separate gate/up (quantized models)
        {
          uint32_t cols = static_cast<uint32_t>(gi.dimensions[0]);
          uint32_t rows = static_cast<uint32_t>(gi.dimensions[1]);
          layer.w_gate = uploadWeightMaybeQuantized(
              reader, blk + "ffn_gate.weight", rows, cols);
        }
        {
          const auto &info = reader.get_tensor_info(blk + "ffn_up.weight");
          uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
          uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
          layer.w_up = uploadWeightMaybeQuantized(reader, blk + "ffn_up.weight",
                                                  rows, cols);
        }
      }

      // ffn_down always separate (not fused)
      {
        const auto &info = reader.get_tensor_info(blk + "ffn_down.weight");
        uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
        uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
        layer.w_down = uploadWeightMaybeQuantized(
            reader, blk + "ffn_down.weight", rows, cols);
      }
    }
  }
  {
    auto layersEnd = std::chrono::high_resolution_clock::now();
    double layersMs =
        std::chrono::duration<double, std::milli>(layersEnd - layersStart)
            .count();
    std::cout << "  Loaded all " << config_.n_layers << " layers in "
              << layersMs << " ms\n";
  }

  // Output norm
  {
    auto data = reader.read_tensor_f32("output_norm.weight");
    output_norm_ = uploadVector(data);
  }

  // Output weight (LM head)
  if (reader.has_tensor("output.weight")) {
    const auto &info = reader.get_tensor_info("output.weight");
    uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
    uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
    output_weight_ =
        uploadWeightMaybeQuantized(reader, "output.weight", rows, cols);
  } else {
    // Some models tie embeddings — use token_embd.weight transposed
    output_weight_.handle = ops_->transpose(embeddingTable_);
  }

  // Initialize KV caches with pre-allocated GPU buffers.
  // Float16 KV cache halves memory (matches llama.cpp default) while
  // attention computation stays Float32 for numerical stability.
  kv_caches_.resize(config_.n_layers);
  for (auto &cache : kv_caches_) {
    cache.k_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float16);
    cache.v_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float16);
  }

  // Pre-allocate buffers for command buffer reuse.
  // Use mapped (host-visible) memory for small per-token params to avoid
  // staging command buffer + fence wait overhead on every token.
  runtimeParamsBuffer_ =
      runtime_->createTensorMapped({2}, cut::DataType::UInt32);
  tokenIdBuffer_ = runtime_->createTensorMapped({1}, cut::DataType::UInt32);
  hiddenBuffer_ =
      runtime_->createTensorEmpty({config_.dim}, cut::DataType::Float32);
  ropeQOutBuffer_ = runtime_->createTensorEmpty(
      {config_.n_heads * config_.head_dim}, cut::DataType::Float32);
  ropeKOutBuffer_ =
      runtime_->createTensorEmpty({config_.kv_dim}, cut::DataType::Float32);
  attnOutBuffer_ = runtime_->createTensorEmpty(
      {config_.n_heads * config_.head_dim}, cut::DataType::Float32);
  // Initialize penalty factors to 1.0 (no penalty) for the first forward pass
  {
    std::vector<float> ones(config_.vocab_size, 1.0f);
    penaltyFactorsBuffer_ = runtime_->createTensorMapped(
        {1, config_.vocab_size}, cut::DataType::Float32, ones.data());
  }
  argmaxResultBuffer_ =
      runtime_->createTensorEmpty({1}, cut::DataType::Float32);

  // Precompute RoPE tables
  precomputeRoPE();

  // Build and optimize graph templates for forward pass
  auto graphStart = std::chrono::high_resolution_clock::now();
  buildGraphTemplates();
  auto graphEnd = std::chrono::high_resolution_clock::now();

  runtime_->flush();

  // Release buffer cache and staging memory — all weight uploads are done.
  runtime_->releaseLoadingResources();

  auto flushEnd = std::chrono::high_resolution_clock::now();
  {
    double graphMs =
        std::chrono::duration<double, std::milli>(graphEnd - graphStart)
            .count();
    double flushMs =
        std::chrono::duration<double, std::milli>(flushEnd - graphEnd).count();
    double totalMs =
        std::chrono::duration<double, std::milli>(flushEnd - loadStart).count();
    std::cout << "Graph build: " << graphMs << " ms, flush: " << flushMs
              << " ms, total load: " << totalMs << " ms\n";
  }
  std::cout << "Model loaded successfully. Buffers: " << runtime_->bufferCount()
            << "  GPU memory: "
            << (runtime_->activeBufferMemoryBytes() / (1024.0 * 1024.0))
            << " MB\n";

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
    std::vector<cut::graph::NamedGraph> graphs;
    if (!layerGraphs_.empty()) {
      auto &lg = layerGraphs_[0];
      graphs.push_back({"QKV Projection", &lg.qkvProjection.preOptGraph,
                        lg.qkvProjection.graph.get(), &lg.qkvProjection.stats});
      graphs.push_back(
          {"Attention Output + Residual", &lg.attnOutputResidual.preOptGraph,
           lg.attnOutputResidual.graph.get(), &lg.attnOutputResidual.stats});
      graphs.push_back({"FFN + Residual", &lg.ffnResidual.preOptGraph,
                        lg.ffnResidual.graph.get(), &lg.ffnResidual.stats});
    }
    graphs.push_back({"Logits", &logitsGraph_.preOptGraph,
                      logitsGraph_.graph.get(), &logitsGraph_.stats});
    generateModelReport(reader, config_, reportPath, graphs);
  }
}

// ============================================================================
// RoPE precomputation
// ============================================================================

void LlamaModel::precomputeRoPE() {
  uint32_t half_dim = config_.head_dim / 2;
  std::vector<float> cos_table(config_.max_seq_len * half_dim);
  std::vector<float> sin_table(config_.max_seq_len * half_dim);

  for (uint32_t pos = 0; pos < config_.max_seq_len; ++pos) {
    for (uint32_t i = 0; i < half_dim; ++i) {
      float freq =
          1.0f / std::pow(config_.rope_freq_base,
                          static_cast<float>(2 * i) / config_.head_dim);
      float angle = static_cast<float>(pos) * freq;
      cos_table[pos * half_dim + i] = std::cos(angle);
      sin_table[pos * half_dim + i] = std::sin(angle);
    }
  }

  // Upload as 1D GPU tensors (shader indexes linearly: pos * halfDim + i)
  rope_cos_gpu_ =
      runtime_->createTensor({config_.max_seq_len * half_dim},
                             cut::DataType::Float32, cos_table.data());
  rope_sin_gpu_ =
      runtime_->createTensor({config_.max_seq_len * half_dim},
                             cut::DataType::Float32, sin_table.data());
}

// ============================================================================
// RMS Norm
// ============================================================================

cut::ComputeHandle LlamaModel::rmsNorm(const cut::ComputeHandle &x,
                                       const cut::ComputeHandle &weight) {
  // x is 1D [dim] — use fused RMSNorm kernel (single dispatch)
  return ops_->rmsNorm(x, weight, config_.norm_eps);
}

// ============================================================================
// RoPE application
// ============================================================================

cut::ComputeHandle
LlamaModel::applyRoPE(const cut::ComputeHandle &x,
                      const cut::ComputeHandle &preallocOutput) {
  return ops_->applyRoPE(x, rope_cos_gpu_, rope_sin_gpu_, runtimeParamsBuffer_,
                         config_.head_dim, preallocOutput);
}

// ============================================================================
// Attention
// ============================================================================

void LlamaModel::attention(const cut::ComputeHandle &q,
                           const cut::ComputeHandle &k,
                           const cut::ComputeHandle &v,
                           int layer) {
  auto &cache = kv_caches_[layer];

  // Write K and V into GPU cache at position from runtimeParamsBuffer_
  ops_->cacheWrite(cache.k_cache, k, runtimeParamsBuffer_);
  ops_->cacheWrite(cache.v_cache, v, runtimeParamsBuffer_);

  // Compute attention — seqLen read from runtimeParamsBuffer_ by shader
  ops_->attention(q, cache.k_cache, cache.v_cache, runtimeParamsBuffer_,
                  config_.n_heads, config_.n_kv_heads, config_.head_dim,
                  attnOutBuffer_);
}

// ============================================================================
// Graph template builders
// ============================================================================

GraphTemplate LlamaModel::buildQKVProjectionGraph(const LlamaLayer &layer) {
  cut::graph::GraphBuilder builder(*runtime_);
  int32_t dim = static_cast<int32_t>(config_.dim);

  // Dynamic input: hidden state [dim] — use ffn_norm as shape placeholder
  // (must differ from attn_norm which is used as constant input below)
  auto vHidden = builder.input(layer.ffn_norm, /*isConstant=*/false);

  // Constant input: attention norm weight [dim]
  auto vAttnNormWeight = builder.input(layer.attn_norm, /*isConstant=*/true);

  // RMS norm inside the graph (saves a standalone dispatch)
  auto vNormed =
      builder.ops().rmsNorm(vHidden, vAttnNormWeight, config_.norm_eps);

  auto normed_2d = builder.ops().reshape(vNormed, {1, dim});

  cut::Tensor q_flat, k_flat, v_flat;

  if (layer.wqkv.handle && !layer.bq) {
    // Fused QKV: single matmul → outputs [qdim + 2*kvdim] combined buffer.
    // The caller splits Q/K/V using createTensorView after graph execution.
    int32_t qdim = static_cast<int32_t>(config_.n_heads * config_.head_dim);
    int32_t kvdim = static_cast<int32_t>(config_.kv_dim);
    int32_t total = qdim + 2 * kvdim;

    auto vQKV = builder.input(layer.wqkv.handle, /*isConstant=*/true);
    auto qkv_out = builder.ops().matmul(normed_2d, vQKV);
    auto qkv_flat = builder.ops().reshape(qkv_out, {total});

    builder.markOutput(qkv_flat);
    auto graph = builder.build();

    GraphTemplate tpl;
    tpl.dynamicInputIds = {graph->nodeId(vHidden)};
    tpl.preOptGraph = graph->clone();
    auto optimizer = cut::graph::GraphOptimizer::createDefault();
    optimizer.optimize(*graph, runtime_->store());
    tpl.stats = optimizer.stats();
    tpl.graph = std::move(graph);
    return tpl;
  } else {
    // Fallback: 3 separate matmuls (quantized, or models with bias)
    auto q = graphWeight(builder, layer.wq, normed_2d);
    q_flat = builder.ops().reshape(
        q, {static_cast<int32_t>(config_.n_heads * config_.head_dim)});
    if (layer.bq) {
      auto vBq = builder.input(layer.bq, /*isConstant=*/true);
      q_flat = builder.ops().binaryOp(cut::BinaryAdd, q_flat, vBq);
    }

    auto k = graphWeight(builder, layer.wk, normed_2d);
    k_flat = builder.ops().reshape(k, {static_cast<int32_t>(config_.kv_dim)});
    if (layer.bk) {
      auto vBk = builder.input(layer.bk, /*isConstant=*/true);
      k_flat = builder.ops().binaryOp(cut::BinaryAdd, k_flat, vBk);
    }

    auto v = graphWeight(builder, layer.wv, normed_2d);
    v_flat = builder.ops().reshape(v, {static_cast<int32_t>(config_.kv_dim)});
    if (layer.bv) {
      auto vBv = builder.input(layer.bv, /*isConstant=*/true);
      v_flat = builder.ops().binaryOp(cut::BinaryAdd, v_flat, vBv);
    }
  }

  builder.markOutput(q_flat);
  builder.markOutput(k_flat);
  builder.markOutput(v_flat);

  auto graph = builder.build();

  GraphTemplate tpl;
  tpl.dynamicInputIds = {graph->nodeId(vHidden)};
  tpl.preOptGraph = graph->clone();
  auto optimizer = cut::graph::GraphOptimizer::createDefault();
  optimizer.optimize(*graph, runtime_->store());
  tpl.stats = optimizer.stats();
  tpl.graph = std::move(graph);
  return tpl;
}

GraphTemplate
LlamaModel::buildAttnOutputResidualGraph(const LlamaLayer &layer) {
  cut::graph::GraphBuilder builder(*runtime_);
  int32_t dim = static_cast<int32_t>(config_.dim);

  // Dynamic inputs — each must use a DIFFERENT placeholder tensor so that
  // Operations can distinguish them during graph construction.
  auto vAttnOut = builder.input(layer.attn_norm, /*isConstant=*/false);
  auto vHidden = builder.input(layer.ffn_norm, /*isConstant=*/false);

  auto attn_2d = builder.ops().reshape(vAttnOut, {1, dim});
  // Reshape hidden to [1, dim] so the add operates in the same shape as the
  // matmul output — this allows MatMulBinaryFusionPass to fuse matmul+add
  // into a single dispatch.
  auto hidden_2d = builder.ops().reshape(vHidden, {1, dim});
  auto proj = graphWeight(builder, layer.wo, attn_2d);
  auto result_2d = builder.ops().binaryOp(cut::BinaryAdd, proj, hidden_2d);
  auto result = builder.ops().reshape(result_2d, {dim});

  builder.markOutput(result);

  auto graph = builder.build();

  GraphTemplate tpl;
  tpl.dynamicInputIds = {graph->nodeId(vAttnOut), graph->nodeId(vHidden)};
  tpl.preOptGraph = graph->clone();
  auto optimizer = cut::graph::GraphOptimizer::createDefault();
  optimizer.optimize(*graph, runtime_->store());
  tpl.stats = optimizer.stats();
  tpl.graph = std::move(graph);
  return tpl;
}

GraphTemplate LlamaModel::buildFFNResidualGraph(const LlamaLayer &layer) {
  cut::graph::GraphBuilder builder(*runtime_);
  int32_t dim = static_cast<int32_t>(config_.dim);

  // Dynamic input: hidden state [dim] — use attn_norm as shape placeholder
  // (must differ from ffn_norm which is used as constant input below)
  auto vHidden = builder.input(layer.attn_norm, /*isConstant=*/false);

  // Constant input: FFN norm weight [dim]
  auto vFfnNormWeight = builder.input(layer.ffn_norm, /*isConstant=*/true);

  // RMS norm inside the graph (saves a standalone dispatch)
  auto vNormed =
      builder.ops().rmsNorm(vHidden, vFfnNormWeight, config_.norm_eps);

  auto x_2d = builder.ops().reshape(vNormed, {1, dim});

  int32_t ffn = static_cast<int32_t>(config_.ffn_dim);

  cut::Tensor gate_up;
  if (layer.w_gate_up.handle) {
    // Fused gate+up: single matmul → [1, 2*ffn_dim], then slice into gate/up.
    auto vGateUp = builder.input(layer.w_gate_up.handle, /*isConstant=*/true);
    auto gate_up_out = builder.ops().matmul(x_2d, vGateUp);
    auto gate_up_flat = builder.ops().reshape(gate_up_out, {2 * ffn});

    // Zero-copy slice into gate [ffn_dim] and up [ffn_dim]
    auto gate_slice =
        builder.ops().slice(gate_up_flat, 0, 0, static_cast<uint32_t>(ffn));
    auto up_slice =
        builder.ops().slice(gate_up_flat, 0, static_cast<uint32_t>(ffn),
                            static_cast<uint32_t>(2 * ffn));

    auto gate_silu = builder.ops().unaryOp(cut::UnarySilu, gate_slice);
    auto gate_up_1d =
        builder.ops().binaryOp(cut::BinaryMul, gate_silu, up_slice);
    gate_up = builder.ops().reshape(gate_up_1d, {1, ffn});
  } else {
    // Fallback: 2 separate matmuls (quantized models)
    auto gate = graphWeight(builder, layer.w_gate, x_2d);
    auto gate_id = builder.ops().reshape(gate, {1, ffn});
    auto up = graphWeight(builder, layer.w_up, x_2d);
    auto up_id = builder.ops().reshape(up, {1, ffn});

    auto gate_silu = builder.ops().unaryOp(cut::UnarySilu, gate_id);
    gate_up = builder.ops().binaryOp(cut::BinaryMul, gate_silu, up_id);
  }

  auto out = graphWeight(builder, layer.w_down, gate_up);
  // Reshape hidden to [1, dim] so the add operates in the same shape as the
  // matmul output — enables MatMulBinaryFusionPass to fuse matmul+add.
  auto hidden_2d = builder.ops().reshape(vHidden, {1, dim});
  auto result_2d = builder.ops().binaryOp(cut::BinaryAdd, out, hidden_2d);
  auto result = builder.ops().reshape(result_2d, {dim});

  builder.markOutput(result);

  auto graph = builder.build();

  GraphTemplate tpl;
  tpl.dynamicInputIds = {graph->nodeId(vHidden)};
  tpl.preOptGraph = graph->clone();
  auto optimizer = cut::graph::GraphOptimizer::createDefault();
  optimizer.optimize(*graph, runtime_->store());
  tpl.stats = optimizer.stats();
  tpl.graph = std::move(graph);
  return tpl;
}

GraphTemplate LlamaModel::buildLogitsGraph() {
  cut::graph::GraphBuilder builder(*runtime_);
  int32_t dim = static_cast<int32_t>(config_.dim);

  // Dynamic input: hidden state [dim] — use layers_[0].attn_norm as shape
  // placeholder (must differ from output_norm_ used as constant input below)
  auto vHidden = builder.input(layers_[0].attn_norm, /*isConstant=*/false);

  // Constant input: output norm weight [dim]
  auto vOutputNorm = builder.input(output_norm_, /*isConstant=*/true);

  // RMS norm inside the graph (eliminates standalone dispatch)
  auto vNormed = builder.ops().rmsNorm(vHidden, vOutputNorm, config_.norm_eps);

  auto hidden_2d = builder.ops().reshape(vNormed, {1, dim});
  auto logits_raw = graphWeight(builder, output_weight_, hidden_2d);

  // Dead code: identity reshape (optimizer: IdentityReshapePass + DeadCodePass)
  builder.ops().reshape(logits_raw,
                        {1, static_cast<int32_t>(config_.vocab_size)});

  // Dead code: unused transpose (optimizer: DeadCodePass removes)
  builder.ops().transpose(logits_raw);

  builder.markOutput(logits_raw);

  auto graph = builder.build();

  GraphTemplate tpl;
  tpl.dynamicInputIds = {graph->nodeId(vHidden)};
  tpl.preOptGraph = graph->clone();
  auto optimizer = cut::graph::GraphOptimizer::createDefault();
  optimizer.optimize(*graph, runtime_->store());
  tpl.stats = optimizer.stats();
  tpl.graph = std::move(graph);
  return tpl;
}

void LlamaModel::buildGraphTemplates() {
  executor_ =
      std::make_unique<cut::graph::GraphExecutor>(*ops_, runtime_->store());

  // Collect optimization statistics across all graph templates
  std::map<std::string, int> totalOptimizations;

  // Build layer 0 graphs and collect stats (representative)
  if (config_.n_layers > 0) {
    layerGraphs_.resize(config_.n_layers);

    auto qkvTpl = buildQKVProjectionGraph(layers_[0]);
    for (const auto &stat : qkvTpl.stats) {
      totalOptimizations[stat.name] += stat.runCount;
    }
    layerGraphs_[0].qkvProjection = std::move(qkvTpl);

    auto attnTpl = buildAttnOutputResidualGraph(layers_[0]);
    for (const auto &stat : attnTpl.stats) {
      totalOptimizations[stat.name] += stat.runCount;
    }
    layerGraphs_[0].attnOutputResidual = std::move(attnTpl);

    auto ffnTpl = buildFFNResidualGraph(layers_[0]);
    for (const auto &stat : ffnTpl.stats) {
      totalOptimizations[stat.name] += stat.runCount;
    }
    layerGraphs_[0].ffnResidual = std::move(ffnTpl);

    // Build remaining layers (reuse same patterns, multiply stats)
    for (uint32_t i = 1; i < config_.n_layers; ++i) {
      layerGraphs_[i].qkvProjection = buildQKVProjectionGraph(layers_[i]);
      layerGraphs_[i].attnOutputResidual =
          buildAttnOutputResidualGraph(layers_[i]);
      layerGraphs_[i].ffnResidual = buildFFNResidualGraph(layers_[i]);
    }

    // Multiply layer 0 stats by n_layers - 1 to account for other layers
    if (config_.n_layers > 1) {
      for (auto &[name, count] : totalOptimizations) {
        count *= config_.n_layers;
      }
    }
  }

  // Build logits graph
  logitsGraph_ = buildLogitsGraph();
  for (const auto &stat : logitsGraph_.stats) {
    totalOptimizations[stat.name] += stat.runCount;
  }

  std::cout << "Built and optimized " << (config_.n_layers * 3 + 1)
            << " graph templates.\n";

  // Print optimization summary
  std::cout << "\nOptimization summary:\n";
  int totalPasses = 0;
  for (const auto &[name, count] : totalOptimizations) {
    if (count > 0) {
      std::cout << "  " << name << ": " << count << " optimizations\n";
      totalPasses += count;
    }
  }
  if (totalPasses == 0) {
    std::cout << "  No optimizations applied (graphs already optimal)\n";
  } else {
    std::cout << "  Total: " << totalPasses
              << " optimizations across all graphs\n";
  }
  std::cout << "\n";
}

std::vector<cut::Tensor> LlamaModel::executeGraph(
    GraphTemplate &tpl, const std::vector<cut::ComputeHandle> &dynamicHandles) {
  for (size_t i = 0; i < tpl.dynamicInputIds.size(); ++i) {
    auto &gn = tpl.graph->node(tpl.dynamicInputIds[i]);
    static_cast<cut::InputOpNode *>(gn.op.get())
        ->setGpuHandle(dynamicHandles[i]);
  }
  return executor_->execute(*tpl.graph);
}

void LlamaModel::splitQKV(const std::vector<cut::Tensor> &qkv,
                          cut::ComputeHandle &q,
                          cut::ComputeHandle &k,
                          cut::ComputeHandle &v) {
  if (qkv.size() == 1) {
    // Fused QKV: split the combined [qdim + 2*kvdim] buffer into Q, K, V
    // views. Explicit barrier needed because the barrier tracker doesn't see
    // views as sharing the matmul's parent buffer.
    uint32_t qdim = config_.n_heads * config_.head_dim;
    uint32_t kvdim = config_.kv_dim;
    q = runtime_->store().createTensorView(qkv[0], 0, {qdim},
                                           cut::DataType::Float32);
    k = runtime_->store().createTensorView(qkv[0], qdim * sizeof(float),
                                           {kvdim}, cut::DataType::Float32);
    v = runtime_->store().createTensorView(qkv[0],
                                           (qdim + kvdim) * sizeof(float),
                                           {kvdim}, cut::DataType::Float32);
    ops_->barrier();
  } else {
    q = qkv[0];
    k = qkv[1];
    v = qkv[2];
  }
}

cut::ComputeHandle LlamaModel::runLayer(uint32_t layerIdx,
                                        const cut::ComputeHandle &hidden) {
  auto &lg = layerGraphs_[layerIdx];

  auto qkv = executeGraph(lg.qkvProjection, {hidden});

  cut::ComputeHandle q, k, v;
  splitQKV(qkv, q, k, v);

  auto &cache = kv_caches_[layerIdx];
  ops_->fusedAttention(q, k, v, cache.k_cache, cache.v_cache,
                       runtimeParamsBuffer_, rope_cos_gpu_, rope_sin_gpu_,
                       config_.n_heads, config_.n_kv_heads, config_.head_dim,
                       attnOutBuffer_);

  auto attn_result =
      executeGraph(lg.attnOutputResidual, {attnOutBuffer_, hidden});
  auto ffn_result = executeGraph(lg.ffnResidual, {attn_result[0]});
  return ffn_result[0];
}

// ============================================================================
// Forward pass
// ============================================================================

cut::ComputeHandle LlamaModel::forward(int token_id, int pos) {
  uint32_t upos = static_cast<uint32_t>(pos);

  // Update runtime params buffer: {pos, seqLen}
  uint32_t params[2] = {upos, upos + 1};
  runtime_->copyToTensor(runtimeParamsBuffer_, params, sizeof(params));

  // Update token ID for GPU embedding lookup
  uint32_t tid = static_cast<uint32_t>(token_id);
  runtime_->copyToTensor(tokenIdBuffer_, &tid, sizeof(uint32_t));

  if (decodeCBCached_) {
    // Resubmit cached CB: embedding → layers → logits → penalty → argmax.
    // Mapped buffers (runtimeParams, tokenId, penaltyFactors) were updated
    // via direct memcpy above — no staging fence needed.
    runtime_->resubmitAndWait(cachedDecodeCB_);
    return argmaxResultBuffer_;
  }

  // --- First forward: record all dispatches into one reusable CB ---

  ops_->embedding(tokenIdBuffer_, embeddingTable_, hiddenBuffer_);

  auto hidden = hiddenBuffer_;
  for (uint32_t i = 0; i < config_.n_layers; ++i)
    hidden = runLayer(i, hidden);

  // LM head logits
  auto logit_result = executeGraph(logitsGraph_, {hidden});
  logitsOutput_ = logit_result[0];

  // Include penalty + argmax in the cached CB so sampling needs no extra
  // submit. penaltyFactorsBuffer_ is updated via memcpy before each resubmit.
  auto penalized =
      ops_->repetitionPenalty(logitsOutput_, penaltyFactorsBuffer_);
  argmaxResultBuffer_ = ops_->reduce(cut::ReduceArgmax, penalized);

  // Cache the command buffer for reuse on subsequent tokens
  cachedDecodeCB_ = runtime_->submitReusable();
  decodeCBCached_ = static_cast<bool>(cachedDecodeCB_);

  return argmaxResultBuffer_;
}

void LlamaModel::forwardPrefill(int token_id, int pos) {
  uint32_t upos = static_cast<uint32_t>(pos);
  uint32_t params[2] = {upos, upos + 1};
  runtime_->copyToTensor(runtimeParamsBuffer_, params, sizeof(params));

  uint32_t tid = static_cast<uint32_t>(token_id);
  runtime_->copyToTensor(tokenIdBuffer_, &tid, sizeof(uint32_t));

  if (prefillCBCached_) {
    runtime_->resubmitAndWait(cachedPrefillCB_);
    return;
  }

  // Record prefill CB: embedding → layers (no logits/penalty/argmax)
  ops_->embedding(tokenIdBuffer_, embeddingTable_, hiddenBuffer_);

  auto hidden = hiddenBuffer_;
  for (uint32_t i = 0; i < config_.n_layers; ++i)
    hidden = runLayer(i, hidden);

  cachedPrefillCB_ = runtime_->submitReusable();
  prefillCBCached_ = static_cast<bool>(cachedPrefillCB_);
}

int LlamaModel::prefill(const std::vector<int> &tokens) {
  // Record all prompt tokens into a single command buffer using inline buffer
  // updates (vkCmdUpdateBuffer) for runtimeParams and tokenId. This eliminates
  // N-1 fence waits by batching everything into one CB submission.
  for (size_t i = 0; i < tokens.size(); ++i) {
    uint32_t upos = static_cast<uint32_t>(i);
    uint32_t params[2] = {upos, upos + 1};
    runtime_->updateBufferInline(runtimeParamsBuffer_, params, sizeof(params));

    uint32_t tid = static_cast<uint32_t>(tokens[i]);
    runtime_->updateBufferInline(tokenIdBuffer_, &tid, sizeof(uint32_t));

    ops_->embedding(tokenIdBuffer_, embeddingTable_, hiddenBuffer_);

    auto hidden = hiddenBuffer_;
    for (uint32_t l = 0; l < config_.n_layers; ++l)
      hidden = runLayer(l, hidden);

    // Only compute logits + argmax for the last token
    if (i == tokens.size() - 1) {
      auto logit_result = executeGraph(logitsGraph_, {hidden});
      logitsOutput_ = logit_result[0];

      auto penalized =
          ops_->repetitionPenalty(logitsOutput_, penaltyFactorsBuffer_);
      argmaxResultBuffer_ = ops_->reduce(cut::ReduceArgmax, penalized);
    }
  }

  // Single submit + wait for the entire prefill
  runtime_->flushPendingCommands();

  // Read back the argmax result (4 bytes)
  float best = 0.0f;
  runtime_->copyFromTensor(argmaxResultBuffer_, &best, sizeof(float));
  return static_cast<int>(best);
}

// ============================================================================
// Generation
// ============================================================================

GenerationResult LlamaModel::generate(const std::vector<int> &prompt_tokens,
                                      int max_new_tokens,
                                      float repeat_penalty,
                                      int repeat_last_n) {
  resetCache();

  GenerationResult result;
  result.promptTokens = static_cast<int>(prompt_tokens.size());

  std::vector<int> tokens = prompt_tokens;
  int next_token = 0;
  uint32_t vocabSize = config_.vocab_size;

  // Upload penalty factors BEFORE forward() so they're part of the cached CB's
  // staging transfers. The cached CB includes: penalty → argmax.
  // When penalty is 1.0, the factors buffer stays all-ones (initialized in
  // load) and the penalty shader is a no-op, so we skip the upload entirely.
  bool hasPenalty = (repeat_penalty != 1.0f);
  auto uploadPenaltyFactors = [&]() {
    if (!hasPenalty)
      return;
    std::vector<float> factors(vocabSize, 1.0f);
    size_t start = 0;
    if (repeat_last_n > 0 &&
        tokens.size() > static_cast<size_t>(repeat_last_n)) {
      start = tokens.size() - static_cast<size_t>(repeat_last_n);
    }
    for (size_t i = start; i < tokens.size(); ++i) {
      int tid = tokens[i];
      if (tid >= 0 && static_cast<uint32_t>(tid) < vocabSize) {
        factors[tid] = repeat_penalty;
      }
    }
    runtime_->copyToTensor(penaltyFactorsBuffer_, factors.data(),
                           vocabSize * sizeof(float));
  };

  // Read argmax result from GPU (4 bytes) after forward completes.
  auto readArgmax = [&](const cut::ComputeHandle &argmaxBuf) -> int {
    float best = 0.0f;
    runtime_->copyFromTensor(argmaxBuf, &best, sizeof(float));
    return static_cast<int>(best);
  };

  // Process prompt tokens (prefill).
  // Non-last tokens use forwardPrefill() which skips logits/penalty/argmax
  // (~25% less GPU work per token) via a separate cached command buffer.
  // Last token uses full forward() to set up the reusable decode CB.
  auto prefillStart = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
    forwardPrefill(prompt_tokens[i], static_cast<int>(i));
  }

  uploadPenaltyFactors();
  auto argmaxBuf =
      forward(prompt_tokens.back(), static_cast<int>(prompt_tokens.size() - 1));
  next_token = readArgmax(argmaxBuf);

  auto prefillEnd = std::chrono::high_resolution_clock::now();
  result.prefillMs =
      std::chrono::duration<double, std::milli>(prefillEnd - prefillStart)
          .count();

  tokens.push_back(next_token);
  std::cout << "Generated token: " << next_token << "\n";

  auto isStopToken = [&](int tok) {
    if (tok == eos_token_id_)
      return true;
    for (int st : stopTokenIds_) {
      if (tok == st)
        return true;
    }
    return false;
  };

  // Stop on EOS from first sampled token
  if (isStopToken(next_token)) {
    result.tokens = std::move(tokens);
    result.generatedTokens = 1;
    result.generateMs = 0.0;
    return result;
  }

  // Autoregressive generation
  auto genStart = std::chrono::high_resolution_clock::now();
  for (int step = 0; step < max_new_tokens - 1; ++step) {
    int pos = static_cast<int>(prompt_tokens.size()) + step;

    // Upload penalty factors before forward (staged, flushed by resubmit)
    uploadPenaltyFactors();

    auto argmaxBuf = forward(next_token, pos);

    next_token = readArgmax(argmaxBuf);
    tokens.push_back(next_token);

    std::cout << "Generated token: " << next_token << "\n";

    if (isStopToken(next_token)) {
      break;
    }
  }
  auto genEnd = std::chrono::high_resolution_clock::now();
  result.generateMs =
      std::chrono::duration<double, std::milli>(genEnd - genStart).count();

  result.tokens = std::move(tokens);
  result.generatedTokens =
      static_cast<int>(result.tokens.size()) - result.promptTokens;
  return result;
}

// ============================================================================
// Tokenization
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
  if (vocab_.empty()) {
    throw std::runtime_error("Tokenizer data not loaded from GGUF");
  }

  // Dispatch based on tokenizer model type
  if (tokenizerModel_ == "gpt2") {
    return tokenizeBPE(text);
  }

  if (scores_.empty()) {
    throw std::runtime_error("SentencePiece scores not loaded from GGUF");
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

  // Convert symbols to token IDs
  std::vector<int> ids;
  if (addBosToken_) {
    ids.push_back(bos_token_id_);
  }
  for (const auto &sym : symbols) {
    auto it = token_to_id_.find(sym);
    if (it != token_to_id_.end()) {
      ids.push_back(it->second);
    }
    // skip unknown symbols (shouldn't happen with byte fallback)
  }

  return ids;
}

// ============================================================================
// GPT-2 BPE Tokenization
// ============================================================================

// Encode a unicode codepoint as UTF-8 and append to string.
static void appendCodepointUTF8(std::string &out, uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

// Build GPT-2 byte ↔ unicode mapping tables.
// Each byte (0-255) maps to a unique unicode codepoint:
//   printable ASCII + some Latin-1 → identity,
//   all others → U+0100+.
void LlamaModel::buildGPT2ByteEncoder() {
  std::vector<bool> identity(256, false);
  for (int i = 33; i <= 126; ++i)
    identity[i] = true;
  for (int i = 161; i <= 172; ++i)
    identity[i] = true;
  for (int i = 174; i <= 255; ++i)
    identity[i] = true;

  int n = 0;
  for (int i = 0; i < 256; ++i) {
    uint32_t cp = identity[i] ? static_cast<uint32_t>(i) : (256 + n++);
    std::string utf8;
    appendCodepointUTF8(utf8, cp);
    byteToUnicode_[i] = utf8;
    unicodeToByte_[utf8] = static_cast<uint8_t>(i);
  }
}

// Simple GPT-2-style pre-tokenizer: splits text at word boundaries.
// Each word keeps its leading space. Punctuation is grouped separately.
static std::vector<std::string> pretokenizeGPT2(const std::string &text) {
  std::vector<std::string> result;
  size_t i = 0;
  while (i < text.size()) {
    std::string token;

    // Check for contraction suffix: 's 't 're 've 'm 'll 'd
    if (text[i] == '\'') {
      token += '\'';
      i++;
      if (i < text.size()) {
        char c = text[i];
        if (c == 's' || c == 't' || c == 'm' || c == 'd') {
          token += c;
          i++;
        } else if (i + 1 < text.size()) {
          std::string pair(text, i, 2);
          if (pair == "re" || pair == "ve" || pair == "ll") {
            token += pair;
            i += 2;
          }
        }
      }
      result.push_back(token);
      continue;
    }

    // Optional leading space
    if (text[i] == ' ') {
      token += ' ';
      i++;
      if (i >= text.size()) {
        result.push_back(token);
        break;
      }
    }

    unsigned char c = static_cast<unsigned char>(text[i]);
    if (std::isalpha(c)) {
      // Collect letters
      while (i < text.size() &&
             std::isalpha(static_cast<unsigned char>(text[i]))) {
        token += text[i++];
      }
    } else if (std::isdigit(c)) {
      // Collect digits
      while (i < text.size() &&
             std::isdigit(static_cast<unsigned char>(text[i]))) {
        token += text[i++];
      }
    } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      // Whitespace (wasn't consumed as leading space above)
      while (i < text.size() && (text[i] == ' ' || text[i] == '\t' ||
                                 text[i] == '\n' || text[i] == '\r')) {
        token += text[i++];
      }
    } else {
      // Other (punctuation, non-ASCII bytes)
      while (i < text.size()) {
        unsigned char ch = static_cast<unsigned char>(text[i]);
        if (std::isalpha(ch) || std::isdigit(ch) || ch == ' ' || ch == '\t' ||
            ch == '\n' || ch == '\r' || ch == '\'') {
          break;
        }
        token += text[i++];
      }
    }

    if (!token.empty()) {
      result.push_back(token);
    }
  }
  return result;
}

// Encode a text segment (no special tokens) using GPT-2 byte-level BPE.
void LlamaModel::encodeBPESegment(const std::string &segment,
                                  std::vector<int> &ids) const {
  auto pretokens = pretokenizeGPT2(segment);

  for (const auto &pt : pretokens) {
    // Convert each byte to its GPT-2 unicode character
    std::string converted;
    for (unsigned char c : pt) {
      converted += byteToUnicode_[c];
    }

    // Split into individual UTF-8 characters
    std::vector<std::string> symbols;
    for (size_t i = 0; i < converted.size();) {
      int len = utf8_len(static_cast<uint8_t>(converted[i]));
      if (i + len > converted.size())
        len = 1;
      symbols.push_back(converted.substr(i, len));
      i += len;
    }

    // BPE merge loop — find the pair with highest priority
    // (lowest rank in mergePriority_) and merge it, repeat.
    while (symbols.size() >= 2) {
      int bestRank = INT_MAX;
      size_t bestIdx = SIZE_MAX;
      for (size_t j = 0; j + 1 < symbols.size(); ++j) {
        std::string pair = symbols[j] + " " + symbols[j + 1];
        auto it = mergePriority_.find(pair);
        if (it != mergePriority_.end() && it->second < bestRank) {
          bestRank = it->second;
          bestIdx = j;
        }
      }
      if (bestIdx == SIZE_MAX)
        break;
      symbols[bestIdx] += symbols[bestIdx + 1];
      symbols.erase(symbols.begin() + bestIdx + 1);
    }

    // Convert symbols to token IDs
    for (const auto &sym : symbols) {
      auto it = token_to_id_.find(sym);
      if (it != token_to_id_.end()) {
        ids.push_back(it->second);
      }
    }
  }
}

std::vector<int> LlamaModel::tokenizeBPE(const std::string &text) const {
  std::vector<int> ids;
  if (addBosToken_) {
    ids.push_back(bos_token_id_);
  }

  // Split text on special token boundaries (e.g. <|im_start|>, <|im_end|>).
  // Special tokens are matched literally and emitted as single token IDs.
  // Text between special tokens is encoded with GPT-2 byte-level BPE.
  size_t pos = 0;
  while (pos < text.size()) {
    // Check if a special token starts at this position
    bool found = false;
    for (const auto &[tok_str, tok_id] : specialTokens_) {
      if (text.compare(pos, tok_str.size(), tok_str) == 0) {
        ids.push_back(tok_id);
        pos += tok_str.size();
        found = true;
        break;
      }
    }
    if (found)
      continue;

    // Find the next special token (or end of text)
    size_t next = text.size();
    for (const auto &[tok_str, tok_id] : specialTokens_) {
      size_t f = text.find(tok_str, pos);
      if (f != std::string::npos && f < next) {
        next = f;
      }
    }

    // Encode the non-special segment with BPE
    if (next > pos) {
      encodeBPESegment(text.substr(pos, next - pos), ids);
    }
    pos = next;
  }

  return ids;
}

std::string LlamaModel::detokenize(const std::vector<int> &tokens) const {
  if (tokenizerModel_ == "gpt2") {
    // GPT-2 BPE: vocab entries use byte-to-unicode encoding.
    // Decode each token's UTF-8 characters back to original bytes.
    std::string result;
    for (int id : tokens) {
      if (id == bos_token_id_ || id == eos_token_id_)
        continue;
      if (id < 0 || static_cast<size_t>(id) >= vocab_.size())
        continue;
      const auto &piece = vocab_[id];
      for (size_t i = 0; i < piece.size();) {
        int len = utf8_len(static_cast<uint8_t>(piece[i]));
        if (i + len > piece.size())
          len = 1;
        std::string ch = piece.substr(i, len);
        auto it = unicodeToByte_.find(ch);
        if (it != unicodeToByte_.end()) {
          result += static_cast<char>(it->second);
        } else {
          result += ch;
        }
        i += len;
      }
    }
    return result;
  }

  // SentencePiece: replace U+2581 (▁) with space
  std::string result;
  for (int id : tokens) {
    if (id >= 0 && static_cast<size_t>(id) < vocab_.size()) {
      const auto &piece = vocab_[id];
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

void LlamaModel::setProfilingEnabled(bool enabled) {
  ops_->setProfilingEnabled(enabled);
}

void LlamaModel::resetCache() {
  for (auto &cache : kv_caches_) {
    // Re-allocate fresh GPU cache buffers (Float16 to match initial allocation)
    cache.k_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float16);
    cache.v_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float16);
    cache.seq_len = 0;
  }
  // Invalidate cached command buffers (new generation = new KV cache handles)
  cachedDecodeCB_.reset();
  decodeCBCached_ = false;
  cachedPrefillCB_.reset();
  prefillCBCached_ = false;
}

} // namespace gguf
