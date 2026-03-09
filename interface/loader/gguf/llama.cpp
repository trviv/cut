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

void LlamaModel::load(const std::string &gguf_path, cut::Runtime &runtime) {
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

    // Attention weights — GGUF/GGML dimensions are [cols, rows] (innermost
    // first). For non-quantized: upload as [rows, cols] and transpose on GPU.
    // For Q8_0: CPU-transpose separated Int8 values + F16 scales to [K, N]
    // to match the regular matmul layout convention.
    {
      const auto &info = reader.get_tensor_info(blk + "attn_q.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      layer.wq =
          uploadWeightMaybeQuantized(reader, blk + "attn_q.weight", rows, cols);
    }
    {
      const auto &info = reader.get_tensor_info(blk + "attn_k.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      layer.wk =
          uploadWeightMaybeQuantized(reader, blk + "attn_k.weight", rows, cols);
    }
    {
      const auto &info = reader.get_tensor_info(blk + "attn_v.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      layer.wv =
          uploadWeightMaybeQuantized(reader, blk + "attn_v.weight", rows, cols);
    }
    {
      const auto &info = reader.get_tensor_info(blk + "attn_output.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      layer.wo = uploadWeightMaybeQuantized(reader, blk + "attn_output.weight",
                                            rows, cols);
    }

    // Attention biases (optional, e.g. Qwen2)
    if (reader.has_tensor(blk + "attn_q.bias")) {
      auto data = reader.read_tensor_f32(blk + "attn_q.bias");
      layer.bq = uploadVector(data);
    }
    if (reader.has_tensor(blk + "attn_k.bias")) {
      auto data = reader.read_tensor_f32(blk + "attn_k.bias");
      layer.bk = uploadVector(data);
    }
    if (reader.has_tensor(blk + "attn_v.bias")) {
      auto data = reader.read_tensor_f32(blk + "attn_v.bias");
      layer.bv = uploadVector(data);
    }

    // FFN norm
    {
      auto data = reader.read_tensor_f32(blk + "ffn_norm.weight");
      layer.ffn_norm = uploadVector(data);
    }

    // FFN weights
    {
      const auto &info = reader.get_tensor_info(blk + "ffn_gate.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      layer.w_gate = uploadWeightMaybeQuantized(reader, blk + "ffn_gate.weight",
                                                rows, cols);
    }
    {
      const auto &info = reader.get_tensor_info(blk + "ffn_up.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      layer.w_up =
          uploadWeightMaybeQuantized(reader, blk + "ffn_up.weight", rows, cols);
    }
    {
      const auto &info = reader.get_tensor_info(blk + "ffn_down.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      layer.w_down = uploadWeightMaybeQuantized(reader, blk + "ffn_down.weight",
                                                rows, cols);
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

  // Initialize KV caches with pre-allocated GPU buffers
  kv_caches_.resize(config_.n_layers);
  for (auto &cache : kv_caches_) {
    cache.k_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float32);
    cache.v_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float32);
  }

  // Pre-allocate buffers for command buffer reuse
  runtimeParamsBuffer_ =
      runtime_->createTensorEmpty({2}, cut::DataType::UInt32);
  tokenIdBuffer_ = runtime_->createTensorEmpty({1}, cut::DataType::UInt32);
  hiddenBuffer_ =
      runtime_->createTensorEmpty({config_.dim}, cut::DataType::Float32);
  ropeQOutBuffer_ = runtime_->createTensorEmpty(
      {config_.n_heads * config_.head_dim}, cut::DataType::Float32);
  ropeKOutBuffer_ =
      runtime_->createTensorEmpty({config_.kv_dim}, cut::DataType::Float32);
  attnOutBuffer_ = runtime_->createTensorEmpty(
      {config_.n_heads * config_.head_dim}, cut::DataType::Float32);
  penaltyFactorsBuffer_ = runtime_->createTensorEmpty({1, config_.vocab_size},
                                                      cut::DataType::Float32);

  // Precompute RoPE tables
  precomputeRoPE();

  // Build and optimize graph templates for forward pass
  auto graphStart = std::chrono::high_resolution_clock::now();
  buildGraphTemplates();
  auto graphEnd = std::chrono::high_resolution_clock::now();

  runtime_->flush();
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

  auto q = graphWeight(builder, layer.wq, normed_2d);
  // Identity reshape (optimizer: IdentityReshapePass eliminates)
  auto q_id = builder.ops().reshape(q, {1, dim});
  // Reshape chain with above (optimizer: ReshapeChainPass collapses)
  auto q_flat = builder.ops().reshape(
      q_id, {static_cast<int32_t>(config_.n_heads * config_.head_dim)});
  // Add bias if present (e.g. Qwen2)
  if (layer.bq) {
    auto vBq = builder.input(layer.bq, /*isConstant=*/true);
    q_flat = builder.ops().binaryOp(cut::BinaryAdd, q_flat, vBq);
  }

  auto k = graphWeight(builder, layer.wk, normed_2d);
  auto k_id =
      builder.ops().reshape(k, {1, static_cast<int32_t>(config_.kv_dim)});
  auto k_flat =
      builder.ops().reshape(k_id, {static_cast<int32_t>(config_.kv_dim)});
  if (layer.bk) {
    auto vBk = builder.input(layer.bk, /*isConstant=*/true);
    k_flat = builder.ops().binaryOp(cut::BinaryAdd, k_flat, vBk);
  }

  auto v = graphWeight(builder, layer.wv, normed_2d);
  auto v_id =
      builder.ops().reshape(v, {1, static_cast<int32_t>(config_.kv_dim)});
  auto v_flat =
      builder.ops().reshape(v_id, {static_cast<int32_t>(config_.kv_dim)});
  if (layer.bv) {
    auto vBv = builder.input(layer.bv, /*isConstant=*/true);
    v_flat = builder.ops().binaryOp(cut::BinaryAdd, v_flat, vBv);
  }

  // Dead code: result never used (optimizer: DeadCodePass removes)
  builder.ops().transpose(q);

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
  // Using the same handle for both would cause the residual add to read from
  // the wrong input (attn_out instead of hidden).
  auto vAttnOut = builder.input(layer.attn_norm, /*isConstant=*/false);
  auto vHidden = builder.input(layer.ffn_norm, /*isConstant=*/false);

  auto attn_2d = builder.ops().reshape(vAttnOut, {1, dim});
  auto proj = graphWeight(builder, layer.wo, attn_2d);
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

  auto gate = graphWeight(builder, layer.w_gate, x_2d);
  // Identity reshape (optimizer: IdentityReshapePass eliminates)
  auto gate_id = builder.ops().reshape(gate, {1, ffn});
  auto up = graphWeight(builder, layer.w_up, x_2d);
  // Identity reshape (optimizer: IdentityReshapePass eliminates)
  auto up_id = builder.ops().reshape(up, {1, ffn});

  auto gate_silu = builder.ops().unaryOp(cut::UnarySilu, gate_id);
  auto gate_up = builder.ops().binaryOp(cut::BinaryMul, gate_silu, up_id);

  auto out = graphWeight(builder, layer.w_down, gate_up);
  // Identity reshape + reshape chain (optimizer eliminates both)
  auto out_id = builder.ops().reshape(out, {1, dim});
  auto out_1d = builder.ops().reshape(out_id, {dim});

  // Dead code: unused transpose (optimizer: DeadCodePass removes)
  builder.ops().transpose(gate);

  auto result = builder.ops().binaryOp(cut::BinaryAdd, vHidden, out_1d);

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

// ============================================================================
// Forward pass
// ============================================================================

cut::ComputeHandle LlamaModel::forward(int token_id, int pos) {
  uint32_t dim = config_.dim;
  uint32_t upos = static_cast<uint32_t>(pos);

  // Update runtime params buffer: {pos, seqLen}
  uint32_t params[2] = {upos, upos + 1};
  runtime_->copyToTensor(runtimeParamsBuffer_, params, sizeof(params));

  // Update token ID for GPU embedding lookup
  uint32_t tid = static_cast<uint32_t>(token_id);
  runtime_->copyToTensor(tokenIdBuffer_, &tid, sizeof(uint32_t));

  if (decodeCBCached_) {
    // Re-submit cached command buffer (no re-recording needed)
    runtime_->resubmitAndWait(cachedDecodeCB_);
    return logitsOutput_;
  }

  // First forward: run with pre-allocated buffers, then cache CB
  // GPU embedding lookup → writes directly to hiddenBuffer_
  ops_->embedding(tokenIdBuffer_, embeddingTable_, hiddenBuffer_);
  ops_->flush();

  auto hidden = hiddenBuffer_;

  // Transformer layers
  for (uint32_t i = 0; i < config_.n_layers; ++i) {
    auto &lg = layerGraphs_[i];

    // QKV projection (graph template — includes rmsNorm internally)
    auto qkv = executeGraph(lg.qkvProjection, {hidden});
    auto q = qkv[0]; // [n_heads * head_dim]
    auto k = qkv[1]; // [kv_dim]
    auto v = qkv[2]; // [kv_dim]

    // Apply RoPE (GPU) with pre-allocated output buffers
    q = applyRoPE(q, ropeQOutBuffer_);
    k = applyRoPE(k, ropeKOutBuffer_);

    // Attention with GPU KV cache (writes to attnOutBuffer_)
    attention(q, k, v, i);

    // Output projection + residual (graph template)
    auto attn_result =
        executeGraph(lg.attnOutputResidual, {attnOutBuffer_, hidden});
    hidden = attn_result[0];

    // FFN + residual (graph template — includes rmsNorm internally)
    auto ffn_result = executeGraph(lg.ffnResidual, {hidden});
    hidden = ffn_result[0];
  }

  // LM head logits (graph template — includes rmsNorm internally)
  auto logit_result = executeGraph(logitsGraph_, {hidden});
  logitsOutput_ = logit_result[0];

  // Cache the command buffer for reuse on subsequent tokens
  cachedDecodeCB_ = runtime_->submitReusable();
  decodeCBCached_ = static_cast<bool>(cachedDecodeCB_);

  return logitsOutput_;
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

  // Sample with GPU repetition penalty + GPU argmax.
  // Only copies 4 bytes (argmax result) back from GPU.
  auto sample = [&](const cut::ComputeHandle &logits) -> int {
    if (repeat_penalty == 1.0f || tokens.empty()) {
      auto argmaxTensor = ops_->reduce(cut::ReduceArgmax, logits);
      float best = 0.0f;
      runtime_->copyFromTensor(argmaxTensor, &best, sizeof(float));
      return static_cast<int>(best);
    }

    // Build penalty factors on CPU: 1.0 = no penalty, >1.0 = penalize
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

    // Upload factors to pre-allocated GPU buffer
    runtime_->copyToTensor(penaltyFactorsBuffer_, factors.data(),
                           vocabSize * sizeof(float));

    // GPU penalty + GPU argmax (graph-recorded, flushed by copyFromTensor)
    auto penalized = ops_->repetitionPenalty(logits, penaltyFactorsBuffer_);
    auto argmaxTensor = ops_->reduce(cut::ReduceArgmax, penalized);
    float best = 0.0f;
    runtime_->copyFromTensor(argmaxTensor, &best, sizeof(float));
    return static_cast<int>(best);
  };

  // Process prompt tokens (prefill)
  auto prefillStart = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < prompt_tokens.size(); ++i) {
    auto logits = forward(prompt_tokens[i], static_cast<int>(i));

    // Only sample from last prompt token
    if (i == prompt_tokens.size() - 1) {
      next_token = sample(logits);
    }
  }
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
    auto logits = forward(next_token, pos);

    next_token = sample(logits);
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
    // Re-allocate fresh GPU cache buffers
    cache.k_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float32);
    cache.v_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float32);
    cache.seq_len = 0;
  }
  // Invalidate cached command buffer (new generation = new KV cache handles)
  cachedDecodeCB_.reset();
  decodeCBCached_ = false;
}

} // namespace gguf
