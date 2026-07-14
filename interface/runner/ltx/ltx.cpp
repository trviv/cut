#include "ltx.h"
#include "Operations.h"
#include "TensorStore.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <stdexcept>

namespace ltx {

static uint16_t f32ToF16(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = u & 0x7fffffu;

  if (exp <= 0) {
    return static_cast<uint16_t>(sign); // flush denormals to signed zero
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u); // clamp to infinity
  }

  // Round-to-nearest on the 13 dropped mantissa bits.
  uint32_t m = mantissa >> 13;
  if (mantissa & 0x1000u) {
    ++m;
    if (m == 0x400u) { // mantissa carry into the exponent
      m = 0;
      ++exp;
      if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | m);
}

// Debug: when CUT_LTX_DEBUG is set, copy a tensor back and print value stats.
static void dumpStats(cut::Runtime &rt, const cut::ComputeHandle &t,
                      const char *name, size_t count) {
  if (!std::getenv("CUT_LTX_DEBUG")) return;
  std::vector<float> v(count);
  rt.copyFromTensor(t, v.data(), count * sizeof(float));
  float mn = v[0], mx = v[0];
  double sum = 0.0; size_t nan = 0;
  for (float x : v) {
    if (!std::isfinite(x)) { ++nan; continue; }
    mn = std::min(mn, x); mx = std::max(mx, x); sum += x;
  }
  std::cout << "  [dbg] " << name << ": n=" << count << " nan=" << nan
            << " min=" << mn << " max=" << mx
            << " mean=" << (sum / std::max<size_t>(1, count - nan))
            << " first=" << v[0] << "," << v[1] << "," << v[2] << "\n";
}

ShardedSafeTensors::ShardedSafeTensors(const std::vector<std::string> &paths) {
  for (const auto &path : paths) {
    readers_.push_back(std::make_unique<safetensor::SafeTensorReader>(path));
  }
}

bool ShardedSafeTensors::has(const std::string &name) const {
  for (const auto &reader : readers_) {
    if (reader->has_tensor(name)) {
      return true;
    }
  }
  return false;
}

std::vector<float> ShardedSafeTensors::readF32(const std::string &name) const {
  for (const auto &reader : readers_) {
    if (reader->has_tensor(name)) {
      return reader->read_tensor_f32(name);
    }
  }
  throw std::runtime_error("tensor not found: " + name);
}

std::vector<size_t> ShardedSafeTensors::shape(const std::string &name) const {
  for (const auto &reader : readers_) {
    if (reader->has_tensor(name)) {
      return reader->get_tensor_info(name).shape;
    }
  }
  throw std::runtime_error("tensor not found: " + name);
}

static uint32_t jsonUInt(const std::string &text, const std::string &key,
                         uint32_t fallback) {
  auto pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) return fallback;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return fallback;
  ++pos;
  while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;
  uint32_t v = 0; bool any = false;
  while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
    v = v * 10 + (text[pos] - '0'); ++pos; any = true;
  }
  return any ? v : fallback;
}
void LtxModel::loadConfig(const std::string &modelDir) {
  std::ifstream f(modelDir + "/transformer/config.json");
  if (!f) {
    throw std::runtime_error("Missing transformer/config.json");
  }
  std::string text((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  config_.inChannels = jsonUInt(text, "in_channels", config_.inChannels);
  config_.nLayers = jsonUInt(text, "num_layers", config_.nLayers);
  config_.nHeads = jsonUInt(text, "num_attention_heads", config_.nHeads);
  config_.headDim = jsonUInt(text, "attention_head_dim", config_.headDim);
  config_.captionDim = jsonUInt(text, "caption_channels", config_.captionDim);
  config_.dim = config_.nHeads * config_.headDim;
  config_.ffnDim = 4 * config_.dim; // diffusers FeedForward mult=4
  std::cout << "Config: layers=" << config_.nLayers
            << " dim=" << config_.dim
            << " heads=" << config_.nHeads << "x" << config_.headDim
            << " caption=" << config_.captionDim << "\n";
}
void LtxModel::load(const std::string &modelDir, cut::Runtime &runtime) {
  runtime_ = &runtime;
  loadConfig(modelDir);
  if (const char *mb = std::getenv("CUT_LTX_FLUSH_MB")) {
    flushBudgetBytes_ = static_cast<size_t>(std::strtoull(mb, nullptr, 10)) << 20;
  }
  if (const char *f16 = std::getenv("CUT_LTX_FP16_ACTS")) {
    fp16Acts_ = std::atoi(f16) != 0;
  }
  fp16Acts_ = fp16Acts_ && config_.dim % 16 == 0 && config_.ffnDim % 16 == 0 &&
              config_.inChannels % 16 == 0 && config_.captionDim % 16 == 0;
  if (const char *fa = std::getenv("CUT_LTX_FUSED_ATTN")) {
    fusedAttn_ = std::atoi(fa) != 0;
  }
  fusedAttn_ = fusedAttn_ && config_.headDim <= 128;
  std::vector<std::string> shardPaths;
  for (const auto &entry : std::filesystem::directory_iterator(modelDir + "/transformer")) {
    if (entry.path().extension() == ".safetensors") {
      shardPaths.push_back(entry.path().string());
    }
  }
  std::sort(shardPaths.begin(), shardPaths.end());
  if (shardPaths.empty()) {
    throw std::runtime_error("No safetensors files found in " + modelDir + "/transformer");
  }

  ShardedSafeTensors st(shardPaths);
  std::cout << "Loading LTX transformer: " << shardPaths.size() << " shards" << std::endl;

  // Block placement
  blockDevice_.assign(config_.nLayers, 0);
  if (const char *split = std::getenv("CUT_DEVICE_SPLIT")) {
    std::istringstream iss(split);
    std::string token;
    size_t dev = 0;
    uint32_t l = 0;
    size_t lastListed = 0;
    while (std::getline(iss, token, ',')) {
      uint32_t n = static_cast<uint32_t>(std::stoul(token));
      if (dev >= runtime.deviceCount()) {
        throw std::runtime_error(
            "CUT_DEVICE_SPLIT has more entries than devices");
      }
      for (uint32_t i = 0; i < n && l < config_.nLayers; ++i) {
        blockDevice_[l++] = dev;
      }
      lastListed = dev;
      ++dev;
    }
    // Blocks beyond the listed counts go to the last listed device.
    while (l < config_.nLayers) {
      blockDevice_[l++] = lastListed;
    }
  }
  firstDevice_ = blockDevice_.front();
  lastDevice_ = blockDevice_.back();
  // Unique devices in block order
  devices_.clear();
  for (size_t d : blockDevice_) {
    if (devices_.empty() || devices_.back() != d) {
      devices_.push_back(d);
    }
  }
  if (devices_.size() > 1) {
    std::cout << "Block placement: ";
    for (size_t d : devices_) std::cout << d << " ";
    std::cout << "\n";
  }

  projInW_ = uploadLinearWeightF16(st, "proj_in.weight", firstDevice_);
  projInB_ = uploadVecF32(st, "proj_in.bias", 1.0f, firstDevice_);
  projOutW_ = uploadLinearWeightF16(st, "proj_out.weight", lastDevice_);
  projOutB_ = uploadVecF32(st, "proj_out.bias", 1.0f, lastDevice_);

  capW1_ = uploadLinearWeightF16(st, "caption_projection.linear_1.weight", firstDevice_);
  capB1_ = uploadVecF32(st, "caption_projection.linear_1.bias", 1.0f, firstDevice_);
  capW2_ = uploadLinearWeightF16(st, "caption_projection.linear_2.weight", firstDevice_);
  capB2_ = uploadVecF32(st, "caption_projection.linear_2.bias", 1.0f, firstDevice_);

  onesDev_.resize(runtime.deviceCount());
  modBufDev_.resize(runtime.deviceCount());
  for (size_t d : devices_) {
    std::vector<float> onesVec(config_.dim, 1.0f);
    onesDev_[d] = runtime_->createTensor({config_.dim}, cut::DataType::Float32, onesVec.data(), false, d);
    modBufDev_[d] = runtime_->createTensorEmpty({config_.nLayers * 6 * config_.dim}, cut::DataType::Float32, false, d);
  }
  finalModBuf_ = runtime_->createTensorEmpty({2 * config_.dim}, cut::DataType::Float32, false, lastDevice_);

  tsL1W_ = st.readF32("time_embed.emb.timestep_embedder.linear_1.weight");
  tsL1B_ = st.readF32("time_embed.emb.timestep_embedder.linear_1.bias");
  tsL2W_ = st.readF32("time_embed.emb.timestep_embedder.linear_2.weight");
  tsL2B_ = st.readF32("time_embed.emb.timestep_embedder.linear_2.bias");

  adaW_ = st.readF32("time_embed.linear.weight");
  adaB_ = st.readF32("time_embed.linear.bias");
  finalScaleShiftTable_ = st.readF32("scale_shift_table");

  blocks_.resize(config_.nLayers);
  for (uint32_t i = 0; i < config_.nLayers; ++i) {
    std::string prefix = "transformer_blocks." + std::to_string(i) + ".";
    blocks_[i].self_ = loadAttn(st, prefix + "attn1.", blockDevice_[i]);
    blocks_[i].cross_ = loadAttn(st, prefix + "attn2.", blockDevice_[i]);

    blocks_[i].ffnW1 = uploadLinearWeightF16(st, prefix + "ff.net.0.proj.weight", blockDevice_[i]);
    blocks_[i].ffnB1 = uploadVecF32(st, prefix + "ff.net.0.proj.bias", 1.0f, blockDevice_[i]);
    blocks_[i].ffnW2 = uploadLinearWeightF16(st, prefix + "ff.net.2.weight", blockDevice_[i]);
    blocks_[i].ffnB2 = uploadVecF32(st, prefix + "ff.net.2.bias", 1.0f, blockDevice_[i]);

    blocks_[i].scaleShiftTable = st.readF32(prefix + "scale_shift_table");
    std::cout << "  block " << i << " loaded\r" << std::flush;
  }
  std::cout << std::endl;

  ropeCosDev_.resize(runtime.deviceCount());
  ropeSinDev_.resize(runtime.deviceCount());

  // Free the (pinned host RAM) staging buffers the weight uploads grew —
  // ~20GB for the 13B model — and drain the recycle cache. Without this the
  // kernel OOM-killer takes the process at generation time.
  runtime_->releaseLoadingResources();
}
cut::ComputeHandle LtxModel::uploadLinearWeightF16(const ShardedSafeTensors &st, const std::string &name, size_t deviceId) {
  auto data = st.readF32(name);
  auto shape = st.shape(name);
  uint32_t out = static_cast<uint32_t>(shape[0]);
  uint32_t in = static_cast<uint32_t>(shape[1]);

  std::vector<uint16_t> tmp(in * out);
  for (uint32_t i = 0; i < in; ++i) {
    for (uint32_t o = 0; o < out; ++o) {
      tmp[i * out + o] = f32ToF16(data[o * in + i]);
    }
  }

  return runtime_->createTensor({in, out}, cut::DataType::Float16, tmp.data(), false, deviceId);
}
cut::ComputeHandle LtxModel::uploadVecF32(const ShardedSafeTensors &st, const std::string &name, float scale, size_t deviceId) {
  auto data = st.readF32(name);
  if (scale != 1.0f) {
    for (auto &val : data) {
      val *= scale;
    }
  }
  return runtime_->createTensor({static_cast<uint32_t>(data.size())}, cut::DataType::Float32, data.data(), false, deviceId);
}
AttnWeights LtxModel::loadAttn(const ShardedSafeTensors &st, const std::string &prefix, size_t deviceId) {
  AttnWeights w;
  w.wq = uploadLinearWeightF16(st, prefix + "to_q.weight", deviceId);
  w.wk = uploadLinearWeightF16(st, prefix + "to_k.weight", deviceId);
  w.wv = uploadLinearWeightF16(st, prefix + "to_v.weight", deviceId);
  w.wo = uploadLinearWeightF16(st, prefix + "to_out.0.weight", deviceId);

  w.bq = uploadVecF32(st, prefix + "to_q.bias", 1.0f, deviceId);
  w.bk = uploadVecF32(st, prefix + "to_k.bias", 1.0f, deviceId);
  w.bv = uploadVecF32(st, prefix + "to_v.bias", 1.0f, deviceId);
  w.bo = uploadVecF32(st, prefix + "to_out.0.bias", 1.0f, deviceId);

  w.normQ = uploadVecF32(st, prefix + "norm_q.weight", 1.0f / std::sqrt(static_cast<float>(config_.headDim)), deviceId);
  w.normK = uploadVecF32(st, prefix + "norm_k.weight", 1.0f, deviceId);
  return w;
}
void LtxModel::computeTimestepModulation(float timestep,
                                        std::vector<float> &outMod,
                                        std::vector<float> &outFinal) const {
  const uint32_t dim = config_.dim;
  const uint32_t F = config_.timeFreqDim;
  const uint32_t half = F / 2;

  // Sinusoidal embedding
  std::vector<float> emb256(F);
  for (uint32_t j = 0; j < half; ++j) {
    float freq = std::exp(-std::log(10000.0f) * static_cast<float>(j) / static_cast<float>(half));
    float arg = timestep * freq;
    emb256[j] = std::cos(arg);
    emb256[half + j] = std::sin(arg);
  }

  // First linear layer
  std::vector<float> e1(dim);
  for (uint32_t o = 0; o < dim; ++o) {
    e1[o] = tsL1B_[o];
    for (uint32_t i = 0; i < F; ++i) {
      e1[o] += tsL1W_[o * F + i] * emb256[i];
    }
  }

  // SiLU activation
  for (uint32_t o = 0; o < dim; ++o) {
    e1[o] = e1[o] / (1.0f + std::exp(-e1[o]));
  }

  // Second linear layer
  std::vector<float> embT(dim);
  for (uint32_t o = 0; o < dim; ++o) {
    embT[o] = tsL2B_[o];
    for (uint32_t i = 0; i < dim; ++i) {
      embT[o] += tsL2W_[o * dim + i] * e1[i];
    }
  }

  // SiLU activation for embedded_timestep
  std::vector<float> sEmb = embT;
  for (uint32_t o = 0; o < dim; ++o) {
    sEmb[o] = sEmb[o] / (1.0f + std::exp(-sEmb[o]));
  }

  // AdaLN modulation
  std::vector<float> temb(6 * dim);
  for (uint32_t k = 0; k < 6 * dim; ++k) {
    temb[k] = adaB_[k];
    for (uint32_t i = 0; i < dim; ++i) {
      temb[k] += adaW_[k * dim + i] * sEmb[i];
    }
  }

  // Build output modulation
  outMod.resize(config_.nLayers * 6 * dim);
  for (uint32_t b = 0; b < config_.nLayers; ++b) {
    for (uint32_t r = 0; r < 6; ++r) {
      for (uint32_t c = 0; c < dim; ++c) {
        float v = blocks_[b].scaleShiftTable[r * dim + c] + temb[r * dim + c];
        if (r == 1 || r == 4) v += 1.0f; // scale_msa and scale_mlp rows get "+1"
        outMod[b * 6 * dim + r * dim + c] = v;
      }
    }
  }

  // Final modulation
  outFinal.resize(2 * dim);
  for (uint32_t c = 0; c < dim; ++c) {
    outFinal[c] = finalScaleShiftTable_[c] + embT[c];
    // Both final rows broadcast the SAME embedded_timestep vector (the
    // reference adds embedded_timestep[:, :, None] over the [2, dim] table).
    outFinal[dim + c] = finalScaleShiftTable_[dim + c] + embT[c] + 1.0f;
  }
}

void LtxModel::computeRopeTables(uint32_t latentFrames, uint32_t latentHeight,
                                uint32_t latentWidth, float frameRate,
                                std::vector<float> &cosTbl,
                                std::vector<float> &sinTbl) const {
  const uint32_t dim = config_.dim;
  const uint32_t nFreq = dim / 6;
  const uint32_t pad = dim % 6;

  float latentFrameRate = frameRate / 8.0f;
  float scale0 = (1.0f / latentFrameRate) * 1.0f / static_cast<float>(config_.ropeBaseFrames);
  float scale1 = 32.0f / static_cast<float>(config_.ropeBaseHeight);
  float scale2 = 32.0f / static_cast<float>(config_.ropeBaseWidth);

  std::vector<float> freqs(nFreq);
  for (uint32_t j = 0; j < nFreq; ++j) {
    freqs[j] = std::pow(config_.ropeTheta,
                        static_cast<float>(j) / static_cast<float>(nFreq - 1));
    freqs[j] *= static_cast<float>(M_PI) / 2.0f;
  }

  uint32_t S = latentFrames * latentHeight * latentWidth;
  cosTbl.resize(S * dim);
  sinTbl.resize(S * dim);

  for (uint32_t fi = 0; fi < latentFrames; ++fi) {
    for (uint32_t hi = 0; hi < latentHeight; ++hi) {
      for (uint32_t wi = 0; wi < latentWidth; ++wi) {
        uint32_t t = fi * latentHeight * latentWidth + hi * latentWidth + wi;
        float g0 = static_cast<float>(fi) * scale0;
        float g1 = static_cast<float>(hi) * scale1;
        float g2 = static_cast<float>(wi) * scale2;

        // Pad values
        for (uint32_t p = 0; p < pad; ++p) {
          cosTbl[t * dim + p] = 1.0f;
          sinTbl[t * dim + p] = 0.0f;
        }

        // Angle layout matches the reference's
        // `freqs.transpose(-1, -2).flatten(2)`: FREQUENCY-major with the
        // three axes (frame, height, width) interleaved per frequency —
        // angle index m = j*3 + axis — then each angle repeated twice
        // (repeat_interleave(2)) after the `pad` leading padding columns.
        const float g[3] = {g0, g1, g2};
        for (uint32_t j = 0; j < nFreq; ++j) {
          for (uint32_t axis = 0; axis < 3; ++axis) {
            float angle = freqs[j] * (g[axis] * 2.0f - 1.0f);
            uint32_t idx = pad + 2 * (j * 3 + axis);
            cosTbl[t * dim + idx] = std::cos(angle);
            cosTbl[t * dim + idx + 1] = std::cos(angle);
            sinTbl[t * dim + idx] = std::sin(angle);
            sinTbl[t * dim + idx + 1] = std::sin(angle);
          }
        }
      }
    }
  }
}
std::vector<float> LtxModel::computeSigmas(uint32_t steps, uint32_t videoSeqLen) const {
  std::vector<float> sigmas(steps + 1);
  if (steps == 1) {
    // Single-step case (parity checks): shift/stretch of sigma=1 is the
    // identity, and the stretch formula would divide by zero.
    sigmas[0] = 1.0f;
    sigmas[1] = 0.0f;
    return sigmas;
  }
  for (uint32_t i = 0; i < steps; ++i) {
    sigmas[i] = 1.0f + static_cast<float>(i) *
                           ((1.0f / static_cast<float>(steps)) - 1.0f) /
                           static_cast<float>(steps - 1);
  }
  sigmas[steps] = 0.0f;

  float m = (config_.maxShift - config_.baseShift) /
            static_cast<float>(config_.maxImageSeqLen - config_.baseImageSeqLen);
  float b = config_.baseShift - m * static_cast<float>(config_.baseImageSeqLen);
  float mu = static_cast<float>(videoSeqLen) * m + b;

  // Exponential dynamic time shift, then stretch so the final sigma equals
  // shift_terminal (FlowMatchEulerDiscreteScheduler with use_dynamic_shifting
  // + shift_terminal). The stretch factor comes from the LAST shifted sigma.
  const float expMu = std::exp(mu);
  for (uint32_t i = 0; i < steps; ++i) {
    sigmas[i] = expMu / (expMu + (1.0f / sigmas[i] - 1.0f));
  }
  const float scaleF =
      (1.0f - sigmas[steps - 1]) / (1.0f - config_.shiftTerminal);
  for (uint32_t i = 0; i < steps; ++i) {
    sigmas[i] = 1.0f - (1.0f - sigmas[i]) / scaleF;
  }

  return sigmas;
}

void LtxModel::releaseTransients() {
  // Submit + wait for all pending GPU work, then drop our references. The
  // handles are refcounted: once the flushed graph's OpNodes and these copies
  // are gone, each buffer is recycled into the backend's size-keyed cache.
  // Holding the references UNTIL the flush is what makes this safe — earlier
  // destruction would let recorded-but-unsubmitted commands alias recycled
  // buffers.
  for (size_t d : devices_) {
    // Operations::flush() releases the graph's OpNode buffer references;
    // flushPendingCommands() alone only submits/waits the command buffer.
    runtime_->ops(d).flush();
    runtime_->flushPendingCommands(d);
  }
  transients_.clear();
  pendingTransientBytes_ = 0;
}

void LtxModel::maybeReleaseTransients() {
  if (pendingTransientBytes_ >= flushBudgetBytes_) {
    releaseTransients();
  }
}

cut::ComputeHandle LtxModel::castAct(cut::Operations &ops,
                                     const cut::ComputeHandle &x,
                                     uint32_t mRows) {
  if (!fp16Acts_ || (mRows % 16) != 0) {
    return x;
  }
  return track(ops.cast(x, cut::DataType::Float16));
}

cut::ComputeHandle LtxModel::mha(const AttnWeights &w, const cut::ComputeHandle &qSrc,
                                const cut::ComputeHandle &kvSrc, uint32_t sq, uint32_t skv,
                                bool useRope, size_t dev) {
  auto &ops = runtime_->ops(dev);

  auto q = track(ops.matmul(castAct(ops, qSrc, sq), w.wq));
  q = track(ops.binaryOpRowBcast(cut::BinaryAdd, q, w.bq));
  q = track(ops.rmsNorm(q, w.normQ, config_.qkNormEps));

  auto k = track(ops.matmul(castAct(ops, kvSrc, skv), w.wk));
  k = track(ops.binaryOpRowBcast(cut::BinaryAdd, k, w.bk));
  k = track(ops.rmsNorm(k, w.normK, config_.qkNormEps));

  if (useRope) {
    q = track(ops.applyRoPEInterleaved(q, ropeCosDev_[dev], ropeSinDev_[dev]));
    k = track(ops.applyRoPEInterleaved(k, ropeCosDev_[dev], ropeSinDev_[dev]));
  }

  auto v = track(ops.matmul(castAct(ops, kvSrc, skv), w.wv));
  v = track(ops.binaryOpRowBcast(cut::BinaryAdd, v, w.bv));

  if (fusedAttn_) {
    // normQ is pre-scaled by 1/sqrt(headDim), so the attention scale is 1.
    auto attnOut = track(ops.ditAttention(q, k, v, config_.nHeads,
                                          config_.headDim, 1.0f));
    auto out = track(ops.matmul(castAct(ops, attnOut, sq), w.wo));
    pendingTransientBytes_ +=
        static_cast<size_t>(sq) * config_.dim * sizeof(float) * 2;
    return track(ops.binaryOpRowBcast(cut::BinaryAdd, out, w.bo));
  }

  auto qT = track(ops.transpose(q));
  auto kT = track(ops.transpose(k));
  auto vT = track(ops.transpose(v));
  ops.barrier();

  uint32_t alignedSq = (sq + 3) & ~3u;
  uint32_t alignedSkv = (skv + 3) & ~3u;
  uint32_t Dh = config_.headDim;
  uint32_t D = config_.dim;

  cut::ComputeHandle acc;
  for (uint32_t h = 0; h < config_.nHeads; ++h) {
    auto qhT = runtime_->store(dev).createTensorView(qT, static_cast<size_t>(h) * Dh * alignedSq * sizeof(float),
                                                  {Dh, sq}, cut::DataType::Float32);
    auto khT = runtime_->store(dev).createTensorView(kT, static_cast<size_t>(h) * Dh * alignedSkv * sizeof(float),
                                                  {Dh, skv}, cut::DataType::Float32);
    auto vhT = runtime_->store(dev).createTensorView(vT, static_cast<size_t>(h) * Dh * alignedSkv * sizeof(float),
                                                  {Dh, skv}, cut::DataType::Float32);

    auto Qh = ops.transpose(qhT);
    auto scores = ops.matmul(Qh, khT);
    auto probs = ops.softmaxFused(scores, 1);
    auto Vh = ops.transpose(vhT);
    auto outH = ops.matmul(probs, Vh);

    auto woH = runtime_->store(dev).createTensorView(w.wo, static_cast<size_t>(h) * Dh * D * sizeof(uint16_t),
                                                  {Dh, D}, cut::DataType::Float16);

    if (h == 0) {
      acc = ops.matmul(outH, woH);
    } else {
      acc = ops.matmulBinary(cut::BinaryAdd, outH, woH, acc);
    }

    // Bound transient VRAM: at large S the heads' score/prob matrix pairs
    // ([sq, skv] each) would otherwise all stay alive until the block ends.
    // Once the accumulated estimate crosses the budget, flushing lets the
    // dead iterations' buffers recycle; `acc` and the tracked q/k/v tensors
    // survive (their handles are still held).
    pendingTransientBytes_ += (static_cast<size_t>(alignedSq) * alignedSkv * 2 +
                               static_cast<size_t>(alignedSq) * Dh * 2) * sizeof(float);
    if (pendingTransientBytes_ >= flushBudgetBytes_) {
      runtime_->ops(dev).flush();
      runtime_->flushPendingCommands(dev);
      pendingTransientBytes_ = 0;
    }
  }

  ops.barrier();
  return track(ops.binaryOpRowBcast(cut::BinaryAdd, acc, w.bo));
}

cut::ComputeHandle LtxModel::block(const BlockWeights &bw, const cut::ComputeHandle &hidden,
                                  const cut::ComputeHandle &encoder, uint32_t sVideo,
                                  uint32_t sText, uint32_t blockIdx) {
  const size_t dev = blockDevice_[blockIdx];
  auto &ops = runtime_->ops(dev);
  const uint32_t D = config_.dim;

  size_t base = static_cast<size_t>(blockIdx) * 6 * D * sizeof(float);
  // Zero-copy views into the per-step modulation buffer (plain locals; the
  // parent buffer is permanent so these are never tracked).
  auto shiftMsa = runtime_->store(dev).createTensorView(modBufDev_[dev], base, {D}, cut::DataType::Float32);
  auto scaleMsa1p = runtime_->store(dev).createTensorView(modBufDev_[dev], base + D * sizeof(float), {D}, cut::DataType::Float32);
  auto gateMsa = runtime_->store(dev).createTensorView(modBufDev_[dev], base + 2 * D * sizeof(float), {D}, cut::DataType::Float32);
  auto shiftMlp = runtime_->store(dev).createTensorView(modBufDev_[dev], base + 3 * D * sizeof(float), {D}, cut::DataType::Float32);
  auto scaleMlp1p = runtime_->store(dev).createTensorView(modBufDev_[dev], base + 4 * D * sizeof(float), {D}, cut::DataType::Float32);
  auto gateMlp = runtime_->store(dev).createTensorView(modBufDev_[dev], base + 5 * D * sizeof(float), {D}, cut::DataType::Float32);

  auto n1 = track(ops.rmsNorm(hidden, onesDev_[dev], config_.blockNormEps));
  n1 = track(ops.binaryOpRowBcast(cut::BinaryMul, n1, scaleMsa1p));
  n1 = track(ops.binaryOpRowBcast(cut::BinaryAdd, n1, shiftMsa));

  auto attn = mha(bw.self_, n1, n1, sVideo, sVideo, true, dev);
  auto gated = track(ops.binaryOpRowBcast(cut::BinaryMul, attn, gateMsa));
  auto h2 = ops.binaryOp(cut::BinaryAdd, hidden, gated);   // untracked until after release
  // Phase boundary (budget-gated): when the sync happens, self-attention
  // transients recycle before cross-attention allocates.
  pendingTransientBytes_ += static_cast<size_t>(sVideo) * D * sizeof(float) * 8;
  maybeReleaseTransients();
  transients_.push_back(h2);

  auto attn2 = mha(bw.cross_, h2, encoder, sVideo, sText, false, dev);
  auto h3 = ops.binaryOp(cut::BinaryAdd, h2, attn2);       // untracked until after release
  pendingTransientBytes_ += static_cast<size_t>(sVideo) * D * sizeof(float) * 8;
  maybeReleaseTransients();
  transients_.push_back(h3);

  auto n2 = track(ops.rmsNorm(h3, onesDev_[dev], config_.blockNormEps));
  n2 = track(ops.binaryOpRowBcast(cut::BinaryMul, n2, scaleMlp1p));
  n2 = track(ops.binaryOpRowBcast(cut::BinaryAdd, n2, shiftMlp));

  auto ff = track(ops.matmul(castAct(ops, n2, sVideo), bw.ffnW1));
  ff = track(ops.binaryOpRowBcast(cut::BinaryAdd, ff, bw.ffnB1));
  {
    auto ffAct = ops.unaryOp(cut::UnaryGelu, ff);          // untracked until after release
    // Phase boundary inside the FFN (budget-gated): the two [S, ffnDim]
    // pre-activation buffers recycle before the down-projection allocates.
    pendingTransientBytes_ += static_cast<size_t>(sVideo) * config_.ffnDim * sizeof(float) * 2;
    maybeReleaseTransients();
    transients_.push_back(ffAct);
    ff = ffAct;
  }
  ff = track(ops.matmul(castAct(ops, ff, sVideo), bw.ffnW2));
  ff = track(ops.binaryOpRowBcast(cut::BinaryAdd, ff, bw.ffnB2));

  auto ffGated = track(ops.binaryOpRowBcast(cut::BinaryMul, ff, gateMlp));
  // h3 is consumed by the final residual add AFTER the last releaseTransients
  // inside the block — its tracked copy from transients_.push_back(h3) was
  // dropped by the FFN-phase release, but the local h3 handle keeps the buffer alive.
  return ops.binaryOp(cut::BinaryAdd, h3, ffGated); // NOT tracked: released by forward()
}

cut::ComputeHandle LtxModel::forward(const cut::ComputeHandle &latents, const std::vector<cut::ComputeHandle> &encoderPerDev,
                                    uint32_t sVideo, uint32_t sText) {
  auto &ops = runtime_->ops(firstDevice_);
  auto h0 = track(ops.matmul(castAct(ops, latents, sVideo), projInW_));
  auto h = ops.binaryOpRowBcast(cut::BinaryAdd, h0, projInB_);
  dumpStats(*runtime_, h, "projIn", (size_t)sVideo * config_.dim);

  size_t curDev = firstDevice_;
  for (uint32_t i = 0; i < config_.nLayers; ++i) {
    const size_t dev = blockDevice_[i];
    if (dev != curDev) {
      // Hop the hidden state across devices (host bounce). transferTensor
      // flushes the source device before reading.
      auto moved = runtime_->transferTensor(h, curDev, dev);
      transients_.push_back(h);
      h = moved;
      curDev = dev;
    }
    auto next = block(blocks_[i], h, encoderPerDev[dev], sVideo, sText, i);
    transients_.push_back(h);
    h = next;
    maybeReleaseTransients();
    if (i == 0 || i == config_.nLayers - 1) {
      dumpStats(*runtime_, h, ("block" + std::to_string(i)).c_str(),
                (size_t)sVideo * config_.dim);
    }
  }

  transients_.push_back(h);
  auto &opsLast = runtime_->ops(lastDevice_);
  auto finalShift = track(runtime_->store(lastDevice_).createTensorView(finalModBuf_, 0, {config_.dim}, cut::DataType::Float32));
  auto finalScale1p = track(runtime_->store(lastDevice_).createTensorView(finalModBuf_, config_.dim * sizeof(float),
                                                         {config_.dim}, cut::DataType::Float32));

  auto n = track(opsLast.layerNorm(h, {config_.dim}, nullptr, nullptr, config_.blockNormEps));
  dumpStats(*runtime_, n, "layerNormRaw", (size_t)sVideo * config_.dim);
  dumpStats(*runtime_, finalModBuf_, "finalModBuf", 2 * (size_t)config_.dim);
  n = track(opsLast.binaryOpRowBcast(cut::BinaryMul, n, finalScale1p));
  dumpStats(*runtime_, n, "afterScale", (size_t)sVideo * config_.dim);
  n = track(opsLast.binaryOpRowBcast(cut::BinaryAdd, n, finalShift));
  dumpStats(*runtime_, n, "finalNorm", (size_t)sVideo * config_.dim);

  auto out0 = track(opsLast.matmul(castAct(opsLast, n, sVideo), projOutW_));
  auto out = opsLast.binaryOpRowBcast(cut::BinaryAdd, out0, projOutB_);
  dumpStats(*runtime_, out, "projOut", (size_t)sVideo * config_.inChannels);
  return out;
}

std::vector<float> LtxModel::generate(const std::vector<float> &promptEmbeds,
                                      uint32_t promptTokens,
                                      const std::vector<float> &negativeEmbeds,
                                      uint32_t negativeTokens,
                                      uint32_t latentFrames,
                                      uint32_t latentHeight,
                                      uint32_t latentWidth,
                                      float frameRate,
                                      uint32_t steps,
                                      float guidanceScale,
                                      uint32_t seed,
                                      const std::vector<float> *initLatents) {
  uint32_t S = latentFrames * latentHeight * latentWidth;
  uint32_t C = config_.inChannels;
  bool cfg = guidanceScale > 1.0f && negativeTokens > 0;

  // Compute RoPE tables
  std::vector<float> cosTbl, sinTbl;
  computeRopeTables(latentFrames, latentHeight, latentWidth, frameRate, cosTbl, sinTbl);
  for (size_t d : devices_) {
    ropeCosDev_[d] = runtime_->createTensor({S, config_.dim}, cut::DataType::Float32, cosTbl.data(), false, d);
    ropeSinDev_[d] = runtime_->createTensor({S, config_.dim}, cut::DataType::Float32, sinTbl.data(), false, d);
  }

  // Compute sigma schedule
  auto sigmas = computeSigmas(steps, S);

  // Initialize latents (from file for reproducible parity checks, else RNG)
  std::vector<float> x(S * C);
  if (initLatents != nullptr) {
    if (initLatents->size() != static_cast<size_t>(S) * C) {
      throw std::runtime_error("initLatents size mismatch");
    }
    x = *initLatents;
  } else {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &val : x) {
      val = nd(rng);
    }
  }

  // Process text embeddings
  auto &ops = runtime_->ops(firstDevice_);
  auto embP = track(runtime_->createTensor({promptTokens, config_.captionDim}, cut::DataType::Float32, promptEmbeds.data(), false, firstDevice_));
  auto e1 = track(ops.matmul(castAct(ops, embP, promptTokens), capW1_));
  e1 = track(ops.binaryOpRowBcast(cut::BinaryAdd, e1, capB1_));
  e1 = track(ops.unaryOp(cut::UnaryGelu, e1));
  auto encPos = ops.matmul(castAct(ops, e1, promptTokens), capW2_);
  encPos = ops.binaryOpRowBcast(cut::BinaryAdd, encPos, capB2_);

  cut::ComputeHandle encNeg;
  if (cfg) {
    auto embN = track(runtime_->createTensor({negativeTokens, config_.captionDim}, cut::DataType::Float32, negativeEmbeds.data(), false, firstDevice_));
    auto e2 = track(ops.matmul(castAct(ops, embN, negativeTokens), capW1_));
    e2 = track(ops.binaryOpRowBcast(cut::BinaryAdd, e2, capB1_));
    e2 = track(ops.unaryOp(cut::UnaryGelu, e2));
    encNeg = ops.matmul(castAct(ops, e2, negativeTokens), capW2_);
    encNeg = ops.binaryOpRowBcast(cut::BinaryAdd, encNeg, capB2_);
  }

  // Replicate encoders to all devices
  std::vector<cut::ComputeHandle> encPosDev(runtime_->deviceCount());
  encPosDev[firstDevice_] = encPos;
  std::vector<cut::ComputeHandle> encNegDev(runtime_->deviceCount());
  if (cfg) encNegDev[firstDevice_] = encNeg;
  for (size_t d : devices_) {
    if (d == firstDevice_) continue;
    encPosDev[d] = runtime_->transferTensor(encPos, firstDevice_, d);
    if (cfg) encNegDev[d] = runtime_->transferTensor(encNeg, firstDevice_, d);
  }
  releaseTransients();

  // Denoising loop
  for (uint32_t i = 0; i < steps; ++i) {
    float sigma = sigmas[i];
    float t = sigma * 1000.0f;

    std::vector<float> mod, fin;
    computeTimestepModulation(t, mod, fin);
    for (size_t d : devices_) {
      runtime_->copyToTensor(modBufDev_[d], mod.data(), mod.size() * sizeof(float), 0, 0, d);
    }
    runtime_->copyToTensor(finalModBuf_, fin.data(), fin.size() * sizeof(float), 0, 0, lastDevice_);

    auto xGpu = track(runtime_->createTensor({S, C}, cut::DataType::Float32, x.data(), false, firstDevice_));
    auto predPos = forward(xGpu, encPosDev, S, promptTokens);
    std::vector<float> npPos(S * C);
    runtime_->copyFromTensor(predPos, npPos.data(), S * C * sizeof(float), 0, 0, lastDevice_);

    std::vector<float> np = npPos;
    if (cfg) {
      auto predNeg = forward(xGpu, encNegDev, S, negativeTokens);
      std::vector<float> npNeg(S * C);
      runtime_->copyFromTensor(predNeg, npNeg.data(), S * C * sizeof(float), 0, 0, lastDevice_);
      for (uint32_t j = 0; j < S * C; ++j) {
        np[j] = npNeg[j] + guidanceScale * (npPos[j] - npNeg[j]);
      }
      transients_.push_back(predNeg);
    }

    transients_.push_back(xGpu);
    transients_.push_back(predPos);
    releaseTransients();

    float dsigma = sigmas[i + 1] - sigma;
    for (uint32_t j = 0; j < S * C; ++j) {
      x[j] += dsigma * np[j];
    }

    std::cout << "step " << (i + 1) << "/" << steps << " sigma=" << sigma << std::endl;
  }

  return x;
}

} // namespace ltx
