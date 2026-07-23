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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>

namespace gguf {

// Debug knob: CUT_DISPATCH_PROFILE=1 aggregates the per-dispatch GPU timings
// collected by Runtime::lastDispatchTimings() and prints them grouped by
// dispatch label. Profiling also disables CUDA graph replay, so absolute
// wall-clock shifts — read the table for relative op cost, not throughput.
static bool dispatchProfileEnabled() {
  static const bool on = std::getenv("CUT_DISPATCH_PROFILE") != nullptr;
  return on;
}

// Prints "total  count  avg  share  label" rows sorted by total GPU time
// descending, for every device that recorded timings since the last flush.
static void dumpDispatchProfile(cut::Runtime &rt, const char *tag) {
  for (size_t d = 0; d < rt.deviceCount(); ++d) {
    auto timings = rt.lastDispatchTimings(d);
    if (timings.empty())
      continue;

    std::map<std::string, std::pair<uint32_t, double>> agg;
    double grand = 0.0;
    for (const auto &t : timings) {
      agg[t.label].first++;
      agg[t.label].second += t.gpuMicros;
      grand += t.gpuMicros;
    }

    std::vector<std::pair<std::string, std::pair<uint32_t, double>>> rows;
    for (const auto &p : agg)
      rows.push_back(p);
    std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
      return a.second.second > b.second.second;
    });

    fprintf(stderr,
            "[dispatch-profile %s] dev=%zu dispatches=%zu totalGpu=%.3fms\n",
            tag, d, timings.size(), grand / 1000.0);
    for (const auto &row : rows) {
      double total = row.second.second;
      uint32_t count = row.second.first;
      fprintf(stderr, "  %9.3fms  %5ux  %9.2fus  %5.1f%%  %s\n", total / 1000.0,
              count, total / count, 100.0 * total / grand, row.first.c_str());
    }
  }
}

// Debug knob: CUT_MEM_REPORT=1 prints a category breakdown of the model's GPU
// buffers after load, so the total reported by activeBufferMemoryBytes() can be
// attributed to weights / embeddings / KV cache / everything else.
static bool memReportEnabled() {
  static const bool on = std::getenv("CUT_MEM_REPORT") != nullptr;
  return on;
}

// Allocated bytes behind one handle, or 0 if the handle is unset.
static size_t handleBytes(cut::Runtime &rt, const cut::ComputeHandle &h,
                          size_t deviceId) {
  if (!h)
    return 0;
  return rt.store(deviceId).getTensor(h).size();
}

// Allocated bytes behind a WeightHandle (plain + quantized values + scales).
static size_t weightBytes(cut::Runtime &rt, const WeightHandle &w,
                          size_t deviceId) {
  return handleBytes(rt, w.handle, deviceId) +
         handleBytes(rt, w.qValues, deviceId) +
         handleBytes(rt, w.qScales, deviceId);
}

// ============================================================================
// Automatic model placement
// ============================================================================

/// Automatic layer placement: decides CUT_DEVICE_SPLIT / CUT_HOST_LAYERS for
/// model.load(); explicit user settings always win.
///
/// Without CUT_DEVICE_BUDGET_MB (multi-device only): if the model does not
/// fit in device 0's memory budget (85% of VRAM), split layers across the
/// initialized devices proportionally to their memory, spilling to host RAM
/// when even the combined budget is too small.
///
/// With CUT_DEVICE_BUDGET_MB set, each device's budget is capped:
///   CUT_DEVICE_BUDGET_MB="512"     -> every device capped at 512 MB
///   CUT_DEVICE_BUDGET_MB="512,256" -> per-device caps (unlisted = uncapped)
/// and placement switches to sequential fill: device 0 up to its budget,
/// overflow to device 1, ..., any remainder becomes host-resident layers
/// (compute on the last device, weights in system RAM). This lets a SMALL
/// model exercise the multi-GPU split + host-overflow machinery.
void autoPlaceModel(cut::Runtime &runtime, const std::string &modelPath) {
  if (std::getenv("CUT_DEVICE_SPLIT") || std::getenv("CUT_HOST_LAYERS")) {
    return; // explicit placement wins
  }

  // Parse CUT_DEVICE_BUDGET_MB: comma-separated MB values, or a single value
  // that applies to every device.
  std::vector<uint64_t> capBytes;
  bool budgetKnob = false;
  if (const char *budgetEnv = std::getenv("CUT_DEVICE_BUDGET_MB")) {
    std::string s(budgetEnv);
    size_t start = 0;
    while (start < s.size()) {
      size_t comma = s.find(',', start);
      if (comma == std::string::npos) {
        comma = s.size();
      }
      uint64_t mb = static_cast<uint64_t>(
          std::atoll(s.substr(start, comma - start).c_str()));
      capBytes.push_back(mb * 1024ull * 1024ull);
      start = comma + 1;
    }
    budgetKnob = !capBytes.empty();
    if (capBytes.size() == 1) {
      capBytes.assign(runtime.deviceCount(), capBytes[0]);
    }
  }

  if (!budgetKnob && runtime.deviceCount() <= 1) {
    return;
  }

  const uint64_t modelBytes = std::filesystem::file_size(modelPath);
  if (modelBytes == 0) {
    return;
  }

  // Per-device budgets: 85% of VRAM, capped by the knob where set.
  std::vector<uint64_t> budgets;
  uint64_t totalBudget = 0;
  for (size_t d = 0; d < runtime.deviceCount(); ++d) {
    uint64_t budget = (runtime.deviceTotalMemoryBytes(d) * 85) / 100;
    if (d < capBytes.size()) {
      budget = std::min(budget, capBytes[d]);
    }
    budgets.push_back(budget);
    totalBudget += budget;
  }

  if (!budgetKnob) {
    // Original proportional placement (unchanged behavior).
    if (modelBytes <= budgets[0]) {
      std::cout << "Auto-placement: model fits on device 0 ("
                << modelBytes / (1024 * 1024) << " MB)\n";
      return;
    }

    gguf::GGUFReader reader(modelPath);
    const auto &meta = reader.metadata();
    const std::string arch = meta.architecture();
    const uint32_t nLayers = meta.get_as<uint32_t>(arch + ".block_count", 0);
    if (nLayers == 0 || totalBudget == 0) {
      return;
    }

    // Distribute layers proportionally to each device's budget.
    std::vector<uint32_t> counts(runtime.deviceCount(), 0);
    uint32_t assigned = 0;
    for (size_t d = 0; d < counts.size(); ++d) {
      counts[d] = std::max<uint32_t>(
          1, static_cast<uint32_t>(std::round(
                 static_cast<double>(nLayers) * budgets[d] / totalBudget)));
      assigned += counts[d];
    }
    // Fix rounding drift on the last device (keep every count >= 1).
    while (assigned > nLayers && counts.back() > 1) {
      --counts.back();
      --assigned;
    }
    if (assigned < nLayers) {
      counts.back() += nLayers - assigned;
    }

    std::string splitStr;
    for (size_t d = 0; d < counts.size(); ++d) {
      if (d > 0) {
        splitStr += ",";
      }
      splitStr += std::to_string(counts[d]);
    }
    setenv("CUT_DEVICE_SPLIT", splitStr.c_str(), 1);
    std::cout << "Auto-placement: CUT_DEVICE_SPLIT=" << splitStr << " ("
              << modelBytes / (1024 * 1024) << " MB across "
              << runtime.deviceCount() << " devices)\n";

    if (modelBytes > totalBudget) {
      const double overflow =
          static_cast<double>(modelBytes - totalBudget) / modelBytes;
      uint32_t hostLayers = std::min(
          nLayers - 1,
          static_cast<uint32_t>(std::ceil(overflow * nLayers)));
      const std::string hostStr = std::to_string(nLayers - hostLayers) + "-" +
                                  std::to_string(nLayers - 1);
      setenv("CUT_HOST_LAYERS", hostStr.c_str(), 1);
      std::cout << "Auto-placement: model exceeds combined device memory — "
                << hostLayers << " layers host-resident (CUT_HOST_LAYERS="
                << hostStr << ")\n";
    }
    return;
  }

  // Budget knob set: sequential fill (device 0 -> device 1 -> ... -> host).
  // Runs even for a single device (overflow goes straight to host RAM) and
  // even when the model would fit in real VRAM — the cap is authoritative.
  gguf::GGUFReader reader(modelPath);
  const auto &meta = reader.metadata();
  const std::string arch = meta.architecture();
  const uint32_t nLayers = meta.get_as<uint32_t>(arch + ".block_count", 0);
  if (nLayers == 0) {
    return;
  }

  // Approximate per-layer weight size from the file size. Ignores the
  // embedding/lm_head mass on the first/last device — fine for a test knob.
  const uint64_t layerBytes = std::max<uint64_t>(1, modelBytes / nLayers);

  uint32_t remaining = nLayers;
  std::vector<uint32_t> counts(runtime.deviceCount(), 0);
  for (size_t d = 0; d < counts.size() && remaining > 0; ++d) {
    uint32_t fit = static_cast<uint32_t>(budgets[d] / layerBytes);
    if (d == 0) {
      fit = std::max<uint32_t>(fit, 1); // keep >=1 layer on the primary device
    }
    counts[d] = std::min(remaining, fit);
    remaining -= counts[d];
  }

  const uint32_t hostCount = remaining;
  if (hostCount > 0) {
    // Overflow layers: compute on the last device, weights in system RAM.
    counts.back() += hostCount;
    const std::string hostStr = std::to_string(nLayers - hostCount) + "-" +
                                std::to_string(nLayers - 1);
    setenv("CUT_HOST_LAYERS", hostStr.c_str(), 1);
    std::cout << "Auto-placement (budget): device budgets exceeded — "
              << hostCount << " layers host-resident (CUT_HOST_LAYERS="
              << hostStr << ")\n";
  }

  // Build the split string up to the last non-zero count. Middle zeros are
  // kept — entry position selects the device index.
  size_t lastNonZero = 0;
  for (size_t d = 0; d < counts.size(); ++d) {
    if (counts[d] > 0) {
      lastNonZero = d;
    }
  }
  std::string splitStr;
  for (size_t d = 0; d <= lastNonZero; ++d) {
    if (d > 0) {
      splitStr += ",";
    }
    splitStr += std::to_string(counts[d]);
  }

  if (lastNonZero == 0 && hostCount == 0) {
    std::cout << "Auto-placement (budget): model fits on device 0 under "
                 "CUT_DEVICE_BUDGET_MB\n";
    return;
  }

  setenv("CUT_DEVICE_SPLIT", splitStr.c_str(), 1);
  std::cout << "Auto-placement (budget): CUT_DEVICE_SPLIT=" << splitStr
            << " (" << modelBytes / (1024 * 1024) << " MB, layer ~"
            << layerBytes / (1024 * 1024) << " MB, budgets:";
  for (size_t d = 0; d < budgets.size(); ++d) {
    std::cout << (d > 0 ? ", " : " ") << budgets[d] / (1024 * 1024) << " MB";
  }
  std::cout << ")\n";
}

// ============================================================================
// Helpers
// ============================================================================

cut::Operations &LlamaModel::opsAt(size_t deviceId) const {
  return runtime_->ops(deviceId);
}

bool LlamaModel::isSegmentStart(uint32_t layerIdx) const {
  return layerIdx == 0 || layerDevice_[layerIdx] != layerDevice_[layerIdx - 1];
}

void LlamaModel::computeLayerPlacement() {
  const size_t nDev = runtime_->deviceCount();
  layerDevice_.assign(config_.n_layers, 0);

  // CUT_DEVICE_SPLIT="n0,n1,..." assigns the first n0 layers to device 0,
  // the next n1 to device 1, etc. Layers beyond the listed counts go to the
  // last listed device.
  if (const char *split = std::getenv("CUT_DEVICE_SPLIT")) {
    std::vector<uint32_t> counts;
    std::string s(split);
    size_t start = 0;
    while (start < s.size()) {
      size_t comma = s.find(',', start);
      if (comma == std::string::npos)
        comma = s.size();
      counts.push_back(static_cast<uint32_t>(
          std::atoi(s.substr(start, comma - start).c_str())));
      start = comma + 1;
    }
    if (counts.empty() || counts.size() > nDev) {
      throw std::runtime_error(
          "CUT_DEVICE_SPLIT lists more devices than the runtime has (" +
          std::to_string(counts.size()) + " > " + std::to_string(nDev) + ")");
    }
    uint32_t l = 0;
    for (size_t d = 0; d < counts.size() && l < config_.n_layers; ++d) {
      for (uint32_t c = 0; c < counts[d] && l < config_.n_layers; ++c) {
        layerDevice_[l++] = static_cast<uint32_t>(d);
      }
    }
    while (l < config_.n_layers) {
      layerDevice_[l++] = static_cast<uint32_t>(counts.size() - 1);
    }
  }

  firstDevice_ = layerDevice_.front();
  lastDevice_ = layerDevice_.back();
  segmentStart_.clear();
  for (uint32_t i = 0; i < config_.n_layers; ++i) {
    if (isSegmentStart(i)) {
      segmentStart_.push_back(i);
    }
  }
  if (segmentStart_.size() > 1) {
    std::cout << "Device placement: " << segmentStart_.size()
              << " pipeline segments across devices:";
    for (uint32_t s : segmentStart_) {
      std::cout << " [layer " << s << "+ -> dev " << layerDevice_[s] << "]";
    }
    std::cout << "\n";
  }

  // CUT_HOST_LAYERS="20-29,5" marks layers (indices and inclusive ranges)
  // whose big weight matrices live in host memory instead of VRAM.
  layerHostResident_.assign(config_.n_layers, 0);
  if (const char *hostLayers = std::getenv("CUT_HOST_LAYERS")) {
    std::string s(hostLayers);
    size_t start = 0;
    uint32_t count = 0;
    while (start < s.size()) {
      size_t comma = s.find(',', start);
      if (comma == std::string::npos)
        comma = s.size();
      std::string entry = s.substr(start, comma - start);
      start = comma + 1;
      size_t dash = entry.find('-');
      uint32_t lo = static_cast<uint32_t>(std::atoi(entry.c_str()));
      uint32_t hi = dash == std::string::npos
                        ? lo
                        : static_cast<uint32_t>(
                              std::atoi(entry.substr(dash + 1).c_str()));
      for (uint32_t l = lo; l <= hi && l < config_.n_layers; ++l) {
        if (!layerHostResident_[l]) {
          layerHostResident_[l] = 1;
          ++count;
        }
      }
    }
    if (count > 0) {
      std::cout << "Host-resident layers: " << count
                << " (weights in system RAM, compute on device)\n";
    }
  }
}

cut::ComputeHandle LlamaModel::demoteToHost(const cut::ComputeHandle &t,
                                            size_t deviceId) {
  if (!t) {
    return t;
  }
  // Copy shape/dtype out before any allocation: creating the mapped tensor
  // can grow the buffer container and invalidate the metadata reference.
  const auto &buf = runtime_->getTensor(t, deviceId);
  const std::vector<uint32_t> shape = buf.getShape();
  const cut::DataType dtype = buf.getDtype();
  const size_t bytes = buf.calculateActualSize();

  std::vector<uint8_t> host(bytes);
  runtime_->copyFromTensor(t, host.data(), bytes, 0, 0, deviceId);
  return runtime_->createTensorMapped(shape, dtype, host.data(), deviceId,
                                      /*preferHost=*/true);
}

void LlamaModel::demoteWeight(WeightHandle &wh, size_t deviceId) {
  wh.handle = demoteToHost(wh.handle, deviceId);
  wh.qValues = demoteToHost(wh.qValues, deviceId);
  wh.qScales = demoteToHost(wh.qScales, deviceId);
}

void LlamaModel::demoteLayerWeights(LlamaLayer &layer, size_t deviceId) {
  demoteWeight(layer.wq, deviceId);
  demoteWeight(layer.wk, deviceId);
  demoteWeight(layer.wv, deviceId);
  demoteWeight(layer.wo, deviceId);
  demoteWeight(layer.wqkv, deviceId);
  demoteWeight(layer.w_gate, deviceId);
  demoteWeight(layer.w_up, deviceId);
  demoteWeight(layer.w_down, deviceId);
  demoteWeight(layer.w_gate_up, deviceId);
}

cut::ComputeHandle LlamaModel::uploadVector(const std::vector<float> &data,
                                            size_t deviceId) {
  std::vector<uint32_t> shape = {static_cast<uint32_t>(data.size())};
  return runtime_->createTensor(shape, cut::DataType::Float32, data.data(),
                                false, deviceId);
}

cut::ComputeHandle LlamaModel::uploadMatrix(const float *data,
                                            uint32_t rows,
                                            uint32_t cols,
                                            size_t deviceId) {
  std::vector<uint32_t> shape = {rows, cols};
  return runtime_->createTensor(shape, cut::DataType::Float32, data, false,
                                deviceId);
}

// Host float -> half narrowing (round-to-nearest, subnormals encoded).
// Subnormals must be encoded rather than flushed: quantizeRowsQ8_0 stores the
// Q8_0 block scale d = amax/127 through here, so a block whose amax is below
// 127 * 2^-14 = 0.00775 would get a zero scale and silently zero all 32 of its
// weights. Models with small weight magnitudes (Mistral-7B-v0.3, rms ~0.0014)
// lose enough blocks that way to produce garbage output.
static uint16_t f32ToF16(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = u & 0x7fffffu;
  if (exp <= 0) {
    // Subnormal range: fp16 subnormals encode mantissa16 * 2^-24, and their bit
    // pattern is just sign | mantissa16 (biased exponent field 0). exp <= 0 here
    // so shift is always >= 14. A round that carries into 0x400 lands exactly on
    // the smallest normal (exponent field 1, mantissa 0), which is correct.
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    if (shift > 24) {
      return static_cast<uint16_t>(sign); // too small even for a subnormal
    }
    uint32_t significand = 0x800000u | mantissa; // restore implicit leading 1
    uint32_t half = significand >> shift;
    if ((significand >> (shift - 1)) & 1u) { // round half up on dropped bits
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u); // clamp to infinity
  }
  uint32_t m = mantissa >> 13;
  if (mantissa & 0x1000u) { // round-to-nearest on the 13 dropped bits
    ++m;
    if (m == 0x400u) {
      m = 0;
      ++exp;
      if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | m);
}

// Quantize a row-major f32 array to GGML Q8_0 blocks (32 values per block),
// matching llama.cpp's quantize_row_q8_0_ref: d = amax/127 (stored as f16),
// q = round(x/d). Exact for Q5_0 sources (5-bit q*4 fits int8, d/4 is a
// fp16 exponent shift); K-quant sources round within ~1e-3 relative.
static void quantizeRowsQ8_0(const std::vector<float> &f32,
                             std::vector<int8_t> &values,
                             std::vector<uint16_t> &scales) {
  size_t nBlocks = f32.size() / 32;
  values.resize(f32.size());
  scales.resize(nBlocks);
  for (size_t b = 0; b < nBlocks; ++b) {
    const float *x = f32.data() + b * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::fabs(x[i]));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    scales[b] = f32ToF16(d);
    for (int i = 0; i < 32; ++i)
      values[b * 32 + i] = static_cast<int8_t>(std::lroundf(x[i] * id));
  }
}


// fp16 bit pattern -> float. Inverse of f32ToF16; decodes subnormals rather
// than flushing them (see f32ToF16 above for why that matters for block
// scales) and maps inf/NaN through. Used for the fp16 super-block scales read
// straight out of GGUF K-quant blocks.
static float f16ToF32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x03FFu;
  uint32_t bits;

  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      // 113, not 112: the loop below decrements once for the shift that moves
      // the implicit leading bit into place, so seeding at 112 yields 2^-25
      // for the smallest subnormal instead of 2^-24.
      uint32_t e = 113;
      while ((man & 0x0400u) == 0) {
        man <<= 1;
        --e;
      }
      man &= 0x03FFu;
      bits = sign | (e << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (man << 13);
  }

  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// Unpacks the 6-bit scale/min pair for sub-block j (0..7) of a Q4_K/Q5_K
// super-block. Mirrors get_scale_min_k4 in gguf_reader.cpp, which is file-local
// there.
static void getScaleMinK4(int j, const uint8_t *q, uint8_t &d, uint8_t &m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// Re-encodes a GGUF Q4_K tensor into CUT's affine Q4 layout, already
// transposed into the matmul's [K, N] orientation so no GPU transpose is
// needed. Exact: see the identity above.
//
// GGUF Q4_K super-block = 256 weights in 144 bytes:
//   [d: f16][dmin: f16][scales: 12 bytes of packed 6-bit sc/m pairs][qs: 128]
// Within a super-block, qs is read in 4 groups of 32 bytes; group g supplies
// elements g*64 + l from the low nibble and g*64 + 32 + l from the high
// nibble, using sub-block scales 2*g and 2*g+1 respectively.
//
// Outputs (tightly packed; the runtime pads the innermost dim on upload):
//   packedT [K, N/2]      byte (k, n>>1), low nibble for even n
//   scalesT [2*blocksK, N] rows [0, blocksK) scales, [blocksK, 2*blocksK) mins
//
// Returns false (leaving the outputs untouched) when the shape is not
// re-encodable, so the caller can fall back to the Q8_0 path.
static bool requantizeQ4KToAffine(const std::vector<uint8_t> &raw,
                                  uint32_t N, uint32_t K,
                                  std::vector<uint8_t> &packedT,
                                  std::vector<uint16_t> &scalesT) {
  if (K % 256 != 0 || N % 2 != 0) return false;
  const uint32_t blocksK = K / 32;
  const size_t superPerRow = K / 256;
  const size_t needed = N * superPerRow * 144;
  if (raw.size() < needed) return false;

  packedT.assign(static_cast<size_t>(K) * (N / 2), 0);
  scalesT.assign(static_cast<size_t>(2) * blocksK * N, 0);

  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t s = 0; s < superPerRow; ++s) {
      const uint8_t *block = raw.data() + ((size_t)n * superPerRow + s) * 144;
      uint16_t d, dmin;
      std::memcpy(&d, block, 2);
      std::memcpy(&dmin, block + 2, 2);
      const float fd = f16ToF32(d);
      const float fdmin = f16ToF32(dmin);
      const uint8_t *scales = block + 4;
      const uint8_t *qs = block + 16;

      for (int g = 0; g < 4; ++g) {
        for (int h = 0; h < 2; ++h) {
          const int js = 2 * g + h;
          uint8_t sc, mm;
          getScaleMinK4(js, scales, sc, mm);
          const float scale = fd * sc;
          const float minv = 8.0f * scale - fdmin * mm;
          const uint32_t e0 = s * 256 + g * 64 + h * 32;
          const uint32_t b = e0 / 32;
          scalesT[(size_t)b * N + n] = f32ToF16(scale);
          scalesT[((size_t)blocksK + b) * N + n] = f32ToF16(minv);
          for (uint32_t l = 0; l < 32; ++l) {
            const uint8_t byte = qs[g * 32 + l];
            const uint8_t q = (h == 0) ? (byte & 0xF) : (byte >> 4);
            const uint32_t k = e0 + l;
            packedT[(size_t)k * (N / 2) + (n >> 1)] |= static_cast<uint8_t>(q << ((n & 1) * 4));
          }
        }
      }
    }
  }
  return true;
}

// Weight types that end up as Float16 GPU weights (natively F16, or via the
// GPU-dequant path), and can therefore be concatenated into a fused weight.
// Q8_0/Q4_0 stay quantized (separate values+scales) and are excluded.
static bool isFusableWeightType(GGMLType t) {
  switch (t) {
  case GGMLType::F16:
  case GGMLType::BF16:
  // GPU-dequantized K-quants.
  case GGMLType::Q4_K:
  case GGMLType::Q5_K:
  case GGMLType::Q6_K:
  // CPU-dequantized block types (these also become Float16 weights).
  case GGMLType::Q4_1:
  case GGMLType::Q5_0:
  case GGMLType::Q5_1:
    return true;
  default:
    // Q8_0 / Q4_0 intentionally stay quantized (separate values + scales) and
    // use the quantized matmul path, so they are not fused on their own.
    return false;
  }
}

// Types GGUFReader::read_tensor_f32 can decode, so they can be narrowed to
// Float16 on the host and concatenated into a fused weight. Superset of
// isFusableWeightType: also covers the block types we normally keep quantized.
static bool isDequantizableToF16(GGMLType t) {
  switch (t) {
  case GGMLType::Q8_0:
  case GGMLType::Q4_0:
    return true;
  default:
    return isFusableWeightType(t);
  }
}

// A weight group is fusable when every constituent can be narrowed to fp16 AND
// at least one is already on the fp16 path. That fuses mixed groups (e.g. Q5_0
// q/k with a Q8_0 v, common in "Q4_K_M" GGUFs) while leaving an all-Q8_0/Q4_0
// group on the quantized path so its memory does not double.
// Debug knob: CUT_DISABLE_WEIGHT_FUSION=1 forces the separate (unfused) QKV and
// gate+up weight paths, so the fused weights can be A/B validated.
static bool weightFusionDisabled() {
  static const bool disabled =
      std::getenv("CUT_DISABLE_WEIGHT_FUSION") != nullptr;
  return disabled;
}

// Requantization of non-native quant types to Q8_0 at load. Debug knob:
// CUT_NO_REQUANT=1 restores the old dequantize-to-fp16 behavior.
// Memory budget for the fused gate+up weight, in bytes.
//
// Unlike the fused QKV weight, a fused gate+up does NOT replace the separate
// gate and up weights: prefillBatched cannot split a [N, 2*ffn] matmul result
// into its halves (slice() is 1D and view-only), so it keeps using the separate
// pair and both copies stay resident. The fusion buys one saved decode dispatch
// per layer, which is worth it only while that duplicate is small — measured on
// an RTX 3090, +4.5% decode for 54 MB on SmolLM2-135M, but -10.6% decode and
// +1025 MB on Llama-3.2-1B-f16, where the extra resident bytes cost more than
// the dispatch saves.
//
// CUT_FFN_FUSION_BUDGET_MB overrides the default: 0 disables the fusion
// outright, a large value forces it on.
static uint64_t ffnFusionBudgetBytes() {
  static const uint64_t budget = [] {
    if (const char *env = std::getenv("CUT_FFN_FUSION_BUDGET_MB"))
      return static_cast<uint64_t>(std::strtoull(env, nullptr, 10)) * 1024ull *
             1024ull;
    return 256ull * 1024ull * 1024ull;
  }();
  return budget;
}

static bool requantDisabled() {
  static const bool disabled = std::getenv("CUT_NO_REQUANT") != nullptr;
  return disabled;
}


// Debug knob: CUT_VERIFY_AFFINE_Q4=1 decodes the re-encoded buffers back with
// the same arithmetic the GPU kernel uses and compares against the reader's
// dequantizer, logging the worst absolute and relative error per tensor. The
// re-encode is supposed to be exact up to fp16 storage of scale/min.
static void verifyQ4KAffine(const std::string &name,
                            const std::vector<float> &ref, uint32_t N,
                            uint32_t K, const std::vector<uint8_t> &packedT,
                            const std::vector<uint16_t> &scalesT) {
  const uint32_t blocksK = K / 32;
  double worstAbs = 0.0, worstRel = 0.0;
  uint32_t worstK = 0, worstN = 0;

  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t k = 0; k < K; ++k) {
      const uint8_t byte = packedT[(size_t)k * (N / 2) + (n >> 1)];
      const int nib = (n & 1) ? (byte >> 4) : (byte & 0xF);
      const uint32_t b = k / 32;
      const float scale = f16ToF32(scalesT[(size_t)b * N + n]);
      const float minv = f16ToF32(scalesT[((size_t)blocksK + b) * N + n]);
      const float got = (float(nib) - 8.0f) * scale + minv;
      const float want = ref[(size_t)n * K + k];

      double absErr = std::fabs(got - want);
      double relErr = absErr / std::max(std::fabs(want), 1e-6f);
      if (absErr > worstAbs) {
        worstAbs = absErr;
        worstK = k;
        worstN = n;
      }
      if (relErr > worstRel)
        worstRel = relErr;
    }
  }

  fprintf(stderr,
          "[affine-q4 verify] %s N=%u K=%u worstAbs=%.3e worstRel=%.3e "
          "at k=%u n=%u\n",
          name.c_str(), N, K, worstAbs, worstRel, worstK, worstN);
}

// Debug knob: CUT_NO_AFFINE_Q4=1 sends Q4_K weights back through the Q8_0
// requantization path instead of the exact affine Q4 re-encode. For A/B
// measurement of the 5-bit vs 8.5-bit weight layout.
static bool affineQ4Disabled() {
  static const bool disabled = std::getenv("CUT_NO_AFFINE_Q4") != nullptr;
  return disabled;
}

// Types we requantize to Q8_0 at load (native float types stay fp16 —
// requantizing them would lose precision; Q8_0/Q4_0 are native already).
static bool isRequantizableToQ8(GGMLType t) {
  switch (t) {
  case GGMLType::Q4_1:
  case GGMLType::Q5_0:
  case GGMLType::Q5_1:
  case GGMLType::Q4_K:
  case GGMLType::Q5_K:
  case GGMLType::Q6_K:
    return true;
  default:
    return false;
  }
}

static bool isQ8able(GGMLType t) {
  return t == GGMLType::Q8_0 ||
         (!requantDisabled() && isRequantizableToQ8(t));
}

static bool groupFusable(std::initializer_list<GGMLType> types) {
  bool anyF16Path = false;
  for (GGMLType t : types) {
    if (!isDequantizableToF16(t))
      return false;
    if (isFusableWeightType(t))
      anyF16Path = true;
  }
  return anyF16Path;
}

cut::ComputeHandle
LlamaModel::uploadFusedF16(const GGUFReader &reader,
                           const std::vector<std::string> &names,
                           size_t deviceId) {
  // Read multiple F16 (or GPU-dequantizable) tensors, concatenate rows, upload
  // once, transpose once. Constituent types may differ (a K-quant model mixes
  // e.g. Q4_K and Q6_K) — each is narrowed to fp16 on the host.
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
    const auto &info = reader.get_tensor_info(name);
    if (info.type == GGMLType::F16) {
      auto raw = reader.read_tensor_raw(name);
      memcpy(combined.data() + off, raw.data(), raw.size());
      off += raw.size();
    } else {
      // CPU-dequantize to f32, then narrow to fp16 on the host.
      auto f32 = reader.read_tensor_f32(name);
      auto *dst = reinterpret_cast<uint16_t *>(combined.data() + off);
      for (size_t i = 0; i < f32.size(); ++i)
        dst[i] = f32ToF16(f32[i]);
      off += f32.size() * 2;
    }
  }

  auto gpu = runtime_->createTensor({totalRows, K}, cut::DataType::Float16,
                                    combined.data(), false, deviceId);
  return opsAt(deviceId).transpose(gpu);
}

GGUFReader::Q8SeparatedData LlamaModel::readAsQ8(const GGUFReader &reader,
                                                 const std::string &name) {
  const auto &info = reader.get_tensor_info(name);
  if (info.type == GGMLType::Q8_0)
    return reader.read_tensor_q8_separated(name);
  // Requantize: CPU-dequantize to f32, then quantize to Q8_0 blocks.
  GGUFReader::Q8SeparatedData out;
  auto f32 = reader.read_tensor_f32(name);
  quantizeRowsQ8_0(f32, out.values, out.scales);
  return out;
}

WeightHandle LlamaModel::uploadFusedQ8(const GGUFReader &reader,
                                       const std::vector<std::string> &names,
                                       size_t deviceId) {
  WeightHandle wh;
  // GGUF dims are [cols=K, rows=N]; all inputs share K. Concatenating along
  // rows (N) is contiguous for both the int8 values and the f16 scales, so
  // QKV / gate+up can be fused without dequantizing (no memory increase).
  uint32_t K =
      static_cast<uint32_t>(reader.get_tensor_info(names[0]).dimensions[0]);
  uint32_t blocksK = K / 32;
  uint32_t totalRows = 0;
  for (const auto &name : names)
    totalRows +=
        static_cast<uint32_t>(reader.get_tensor_info(name).dimensions[1]);

  std::vector<int8_t> values;
  std::vector<uint16_t> scales;
  values.reserve(static_cast<size_t>(totalRows) * K);
  scales.reserve(static_cast<size_t>(totalRows) * blocksK);
  for (const auto &name : names) {
    auto q8 = readAsQ8(reader, name);
    values.insert(values.end(), q8.values.begin(), q8.values.end());
    scales.insert(scales.end(), q8.scales.begin(), q8.scales.end());
  }

  auto gpuValues = runtime_->createTensor(
      {totalRows, K}, cut::DataType::Int8, values.data(), false, deviceId);
  auto gpuScales =
      runtime_->createTensor({totalRows, blocksK}, cut::DataType::Float16,
                             scales.data(), false, deviceId);
  wh.qValues = opsAt(deviceId).transpose(gpuValues);
  wh.qScales = opsAt(deviceId).transpose(gpuScales);
  wh.qCols = K;
  wh.quantType = WeightHandle::QuantType::Q8_0;
  return wh;
}

cut::ComputeHandle
LlamaModel::uploadWeight(const GGUFReader &reader,
                         const std::string &name,
                         const std::vector<uint32_t> &shape,
                         size_t deviceId) {
  const auto &info = reader.get_tensor_info(name);

  if (info.type == GGMLType::F16) {
    // Upload Float16 weights directly — no conversion needed.
    auto raw = reader.read_tensor_raw(name);
    return runtime_->createTensor(shape, cut::DataType::Float16, raw.data(),
                                  false, deviceId);
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
          {static_cast<uint32_t>(raw.size())}, cut::DataType::Int8, raw.data(),
          false, deviceId);
      auto deq = opsAt(deviceId).dequantize(
          rawTensor, static_cast<uint32_t>(fmt), shape[0], shape[1]);
      // Store dequantized 2D weights as Float16 (like the native f16 weight
      // path above) to halve decode-time weight bandwidth. K-quant/BF16 values
      // fit fp16 losslessly; transpose + matmul consume fp16 B directly.
      return opsAt(deviceId).cast(deq, cut::DataType::Float16);
    }
  }

  // Fallback: CPU dequantize. 2D weight matrices are narrowed to Float16 (like
  // the GPU-dequant path above) to halve decode-time weight bandwidth; 1D
  // tensors (norms/biases) stay Float32. Many "Q4_K_M" GGUFs store most 2D
  // weights as Q5_0, which lands here rather than in the GPU-dequant switch.
  auto data = reader.read_tensor_f32(name);
  if (shape.size() == 2) {
    std::vector<uint16_t> half(data.size());
    for (size_t i = 0; i < data.size(); ++i)
      half[i] = f32ToF16(data[i]);
    return runtime_->createTensor(shape, cut::DataType::Float16, half.data(),
                                  false, deviceId);
  }
  return runtime_->createTensor(shape, cut::DataType::Float32, data.data(),
                                false, deviceId);
}

WeightHandle LlamaModel::uploadWeightMaybeQuantized(const GGUFReader &reader,
                                                    const std::string &name,
                                                    uint32_t rows,
                                                    uint32_t cols,
                                                    size_t deviceId) {
  const auto &info = reader.get_tensor_info(name);
  WeightHandle wh;

  if (info.type == GGMLType::Q4_0) {
    // Upload Q4_0 weights and GPU-transpose packed nibbles + scales.
    // GGUF layout: [rows=N, cols=K]. Q4 has 2 nibbles per byte.
    auto q4 = reader.read_tensor_q4_separated(name);
    uint32_t N = rows, K = cols;
    uint32_t blocksK = K / 32;

    // Upload packed nibbles [N, K/2] directly to GPU
    auto gpuPacked =
        runtime_->createTensor({N, K / 2}, cut::DataType::Int8,
                               q4.packedValues.data(), false, deviceId);

    // GPU nibble transpose: [N, K/2] -> [K, N/2]
    // Combines unpack (GGML block layout) + transpose + repack in one dispatch
    auto tPacked = opsAt(deviceId).transposeQ4(gpuPacked, N, K);

    // GPU transpose scales [N, K/32] -> [K/32, N]
    auto gpuScales =
        runtime_->createTensor({N, blocksK}, cut::DataType::Float16,
                               q4.scales.data(), false, deviceId);
    auto tScales = opsAt(deviceId).transpose(gpuScales);

    wh.qValues = tPacked;
    wh.qScales = tScales;
    wh.qCols = cols;
    wh.quantType = WeightHandle::QuantType::Q4_0;
    return wh;
  }

  if (info.type == GGMLType::Q4_K && !requantDisabled() && !affineQ4Disabled()) {
    // Exact re-encode into affine Q4 (5 bits/weight) rather than requantizing
    // to Q8_0 (8.5). The re-encoder emits the [K, N] orientation directly, so
    // unlike the Q4_0/Q8_0 branches there is no GPU transpose here.
    const uint32_t N = rows, K = cols;
    std::vector<uint8_t> packedT;
    std::vector<uint16_t> scalesT;
    if (requantizeQ4KToAffine(reader.read_tensor_raw(name), N, K, packedT,
                              scalesT)) {
      static const bool verify =
          std::getenv("CUT_VERIFY_AFFINE_Q4") != nullptr;
      if (verify)
        verifyQ4KAffine(name, reader.read_tensor_f32(name), N, K, packedT,
                        scalesT);
      const uint32_t blocksK = K / 32;
      wh.qValues = runtime_->createTensor({K, N / 2}, cut::DataType::Int8,
                                          packedT.data(), false, deviceId);
      wh.qScales =
          runtime_->createTensor({2 * blocksK, N}, cut::DataType::Float16,
                                 scalesT.data(), false, deviceId);
      wh.qCols = cols;
      // Affine Q4 rides the same kernels as Q4_0; the doubled scales tensor is
      // what switches them into affine mode.
      wh.quantType = WeightHandle::QuantType::Q4_0;
      return wh;
    }
    // Fall through to the Q8_0 path when the shape is not re-encodable.
  }

  if (isQ8able(info.type)) {
    // Upload Q8_0 weights in GGUF-native [N, K] layout, then GPU-transpose
    // to [K, N] to match the matmul layout convention. Non-native quant
    // types (Q5_0, K-quants, ...) are requantized to Q8_0 on the host so
    // they stay on the quantized path instead of doubling to fp16
    // (CUT_NO_REQUANT=1 restores the fp16 path).
    auto q8 = readAsQ8(reader, name);
    uint32_t N = rows, K = cols;
    uint32_t blocksK = K / 32;

    auto gpuValues =
        runtime_->createTensor({N, K}, cut::DataType::Int8, q8.values.data(),
                               false, deviceId);
    auto gpuScales =
        runtime_->createTensor({N, blocksK}, cut::DataType::Float16,
                               q8.scales.data(), false, deviceId);
    wh.qValues = opsAt(deviceId).transpose(gpuValues);
    wh.qScales = opsAt(deviceId).transpose(gpuScales);
    wh.qCols = cols;
    wh.quantType = WeightHandle::QuantType::Q8_0;
    return wh;
  }

  // Non-quantized: upload + transpose as before
  auto gpu = uploadWeight(reader, name, {rows, cols}, deviceId);
  wh.handle = opsAt(deviceId).transpose(gpu);
  return wh;
}

cut::Tensor LlamaModel::embedTokens(const cut::Tensor &indices,
                                    const cut::Tensor &preallocOutput) {
  auto &ops = opsAt(firstDevice_);
  if (embedFromHead_) {
    const auto &mat = output_weight_.isQuantized() ? output_weight_.qValues
                                                   : output_weight_.handle;
    const auto &sc = output_weight_.isQuantized() ? output_weight_.qScales
                                                  : output_weight_.handle;
    return ops.embeddingCol(indices, mat, sc, preallocOutput);
  }
  // preallocOutput selects the shape of the lookup: the single-token paths
  // pass one and reuse the [dim] Float16 staging buffer, batched prefill
  // passes none and needs the op to allocate an [N, dim] result.
  if (preallocOutput) {
    ops.embedding(indices, embeddingTable_, embedOutBuffers_[firstDevice_]);
    return ops.cast(embedOutBuffers_[firstDevice_], cut::DataType::Float32);
  }
  auto half = ops.embedding(indices, embeddingTable_);
  return ops.cast(half, cut::DataType::Float32);
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

  // Head size: prefer explicit attention.key_length (e.g. Mistral-Small has
  // head_dim 128 while dim/n_heads is 160); fall back to dim/n_heads.
  config_.head_dim = meta.get_as<uint32_t>(prefix + "attention.key_length",
                                           config_.dim / config_.n_heads);
  config_.kv_dim = config_.head_dim * config_.n_kv_heads;
  config_.n_rep = config_.n_heads / config_.n_kv_heads;

  // Assign layers to devices (pipeline split via CUT_DEVICE_SPLIT).
  computeLayerPlacement();

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
  // Token embedding source. The table itself is uploaded later, after the LM
  // head is built: for a tied model the head holds the same weights and
  // embeddingCol can gather from it, so the table is never allocated at all.
  {
    const auto &info = reader.get_tensor_info("token_embd.weight");
    config_.vocab_size = static_cast<uint32_t>(info.dimensions[1]);
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
    const size_t dev = layerDevice_[i];

    std::cout << "  Loading layer " << i << "...\r" << std::flush;

    // Attention norm (1D weight)
    {
      auto data = reader.read_tensor_f32(blk + "attn_norm.weight");
      layer.attn_norm = uploadVector(data, dev);
    }

    // Attention biases (optional, e.g. Qwen2)
    bool hasBias = reader.has_tensor(blk + "attn_q.bias");
    if (hasBias) {
      layer.bq = uploadVector(reader.read_tensor_f32(blk + "attn_q.bias"), dev);
      layer.bk = uploadVector(reader.read_tensor_f32(blk + "attn_k.bias"), dev);
      layer.bv = uploadVector(reader.read_tensor_f32(blk + "attn_v.bias"), dev);
    }

    // Attention weights — GGUF/GGML dimensions are [cols, rows] (innermost
    // first). For F16 without bias, build fused QKV directly (single upload +
    // single transpose) instead of uploading Q/K/V separately. This halves
    // attention weight GPU memory vs the old approach.
    {
      const auto &qi = reader.get_tensor_info(blk + "attn_q.weight");
      const auto &ki = reader.get_tensor_info(blk + "attn_k.weight");
      const auto &vi = reader.get_tensor_info(blk + "attn_v.weight");
      // Prefer Q8 fusion whenever every constituent is Q8-able (native Q8_0
      // or requantized): quantized weights read ~2x fewer bytes per token
      // than the fp16-fused path.
      bool canFuseQ8 = !weightFusionDisabled() && !hasBias &&
                       isQ8able(qi.type) && isQ8able(ki.type) &&
                       isQ8able(vi.type);
      bool canFuse = !weightFusionDisabled() && !hasBias && !canFuseQ8 &&
                     groupFusable({qi.type, ki.type, vi.type});

      if (canFuse) {
        layer.wqkv.handle = uploadFusedF16(reader,
                                           {
                                               blk + "attn_q.weight",
                                               blk + "attn_k.weight",
                                               blk + "attn_v.weight",
                                           },
                                           dev);
      } else if (canFuseQ8) {
        layer.wqkv = uploadFusedQ8(reader,
                                   {
                                       blk + "attn_q.weight",
                                       blk + "attn_k.weight",
                                       blk + "attn_v.weight",
                                   },
                                   dev);
      } else {
        // Separate Q/K/V (quantized or biased models)
        uint32_t qCols = static_cast<uint32_t>(qi.dimensions[0]);
        uint32_t qRows = static_cast<uint32_t>(qi.dimensions[1]);
        layer.wq = uploadWeightMaybeQuantized(reader, blk + "attn_q.weight",
                                              qRows, qCols, dev);
        {
          const auto &info = reader.get_tensor_info(blk + "attn_k.weight");
          layer.wk = uploadWeightMaybeQuantized(
              reader, blk + "attn_k.weight",
              static_cast<uint32_t>(info.dimensions[1]),
              static_cast<uint32_t>(info.dimensions[0]), dev);
        }
        {
          const auto &info = reader.get_tensor_info(blk + "attn_v.weight");
          layer.wv = uploadWeightMaybeQuantized(
              reader, blk + "attn_v.weight",
              static_cast<uint32_t>(info.dimensions[1]),
              static_cast<uint32_t>(info.dimensions[0]), dev);
        }
      }
    }

    {
      const auto &info = reader.get_tensor_info(blk + "attn_output.weight");
      uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
      layer.wo = uploadWeightMaybeQuantized(reader, blk + "attn_output.weight",
                                            rows, cols, dev);
    }

    // FFN norm
    {
      auto data = reader.read_tensor_f32(blk + "ffn_norm.weight");
      layer.ffn_norm = uploadVector(data, dev);
    }

    // FFN weights — for F16, build fused gate+up directly (single upload).
    {
      const auto &gi = reader.get_tensor_info(blk + "ffn_gate.weight");
      const auto &ui = reader.get_tensor_info(blk + "ffn_up.weight");
      // Prefer Q8 fusion when the whole group is Q8-able (see QKV above).
      bool canFuseFFNQ8 = !weightFusionDisabled() && isQ8able(gi.type) &&
                          isQ8able(ui.type);
      bool canFuseFFN = !weightFusionDisabled() && !canFuseFFNQ8 &&
                        groupFusable({gi.type, ui.type});

      // The fused weight duplicates gate+up rather than replacing them, so
      // only build it while that duplicate fits the budget. Q8 stores one byte
      // per weight plus an fp16 scale per 32; the f16 path stores two bytes.
      if (canFuseFFNQ8 || canFuseFFN) {
        const uint64_t elems = static_cast<uint64_t>(config_.n_layers) * 2ull *
                               config_.dim * config_.ffn_dim;
        const uint64_t dupBytes =
            canFuseFFNQ8 ? (elems + elems / 32ull * 2ull) : (elems * 2ull);
        if (dupBytes > ffnFusionBudgetBytes()) {
          canFuseFFNQ8 = false;
          canFuseFFN = false;
        }
      }

      if (canFuseFFN) {
        layer.w_gate_up.handle = uploadFusedF16(reader,
                                                {
                                                    blk + "ffn_gate.weight",
                                                    blk + "ffn_up.weight",
                                                },
                                                dev);
      } else if (canFuseFFNQ8) {
        layer.w_gate_up = uploadFusedQ8(reader,
                                        {
                                            blk + "ffn_gate.weight",
                                            blk + "ffn_up.weight",
                                        },
                                        dev);
      }

      // Always load separate w_gate / w_up. Used by:
      //  - prefillBatched (always)
      //  - decode-time runLayer when fused gate_up is unavailable (quantized)
      {
        uint32_t cols = static_cast<uint32_t>(gi.dimensions[0]);
        uint32_t rows = static_cast<uint32_t>(gi.dimensions[1]);
        layer.w_gate = uploadWeightMaybeQuantized(
            reader, blk + "ffn_gate.weight", rows, cols, dev);
      }
      {
        const auto &info = reader.get_tensor_info(blk + "ffn_up.weight");
        uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
        uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
        layer.w_up = uploadWeightMaybeQuantized(reader, blk + "ffn_up.weight",
                                                rows, cols, dev);
      }

      // ffn_down always separate (not fused)
      {
        const auto &info = reader.get_tensor_info(blk + "ffn_down.weight");
        uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
        uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
        layer.w_down = uploadWeightMaybeQuantized(
            reader, blk + "ffn_down.weight", rows, cols, dev);
      }
    }

    // Overflow layers: move the big weight matrices to host memory now that
    // the GPU-side transposes are done.
    if (layerHostResident_[i]) {
      demoteLayerWeights(layer, dev);
    }

    // Flush periodically during bulk loading: staging buffers for pending
    // uploads stay alive until their commands execute, so batching the
    // whole load would roughly double peak memory on large models.
    if ((i + 1) % 4 == 0) {
      runtime_->flush(dev);
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
    output_norm_ = uploadVector(data, lastDevice_);
  }

  // Output weight (LM head)
  if (reader.has_tensor("output.weight")) {
    const auto &info = reader.get_tensor_info("output.weight");
    uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
    uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);
    output_weight_ = uploadWeightMaybeQuantized(reader, "output.weight", rows,
                                                cols, lastDevice_);
  } else {
    const auto &embInfo = reader.get_tensor_info("token_embd.weight");
    static const bool kForceFp16Head =
        std::getenv("CUT_FP16_LM_HEAD") != nullptr;
    // Any type uploadWeightMaybeQuantized can keep quantized: Q4_0 natively,
    // Q4_K as affine Q4, and the rest via Q8_0. Models that tie their
    // embeddings overwhelmingly store token_embd as Q6_K, so restricting this
    // to Q8_0/Q4_0 sent all of them down the fp16 path.
    const bool headQuantizable =
        embInfo.type == GGMLType::Q4_0 || isQ8able(embInfo.type);
    if (!kForceFp16Head && headQuantizable) {
      // Keep the tied head quantized: it is read in full on every decoded
      // token, so this is decode bandwidth as much as resident bytes (419 MB
      // Q8 vs 751 MB fp16 at 128256x3072). graphWeight() already dispatches
      // quantized weights to the 3-input matmul. Uploads straight to
      // lastDevice_, so no transferTensor is needed. CUT_FP16_LM_HEAD=1
      // forces the fp16 path for A/B tests.
      uint32_t cols = static_cast<uint32_t>(embInfo.dimensions[0]);
      uint32_t rows = static_cast<uint32_t>(embInfo.dimensions[1]);
      output_weight_ = uploadWeightMaybeQuantized(reader, "token_embd.weight",
                                                  rows, cols, lastDevice_);
    } else {
      // Tied embeddings with a head type we cannot keep quantized. Upload a
      // temporary fp16 [vocab, dim] copy on lastDevice_ and transpose it; the
      // temporary is released when it goes out of scope, leaving only the
      // [dim, vocab] head. The vocab-projection matmul dominates decode and
      // consumes fp16 B directly.
      auto embd = reader.read_tensor_f32("token_embd.weight");
      std::vector<uint16_t> half(embd.size());
      for (size_t i = 0; i < embd.size(); ++i)
        half[i] = f32ToF16(embd[i]);
      auto embForHead = runtime_->createTensor(
          {config_.vocab_size, config_.dim}, cut::DataType::Float16,
          half.data(), false, lastDevice_);
      output_weight_.handle = opsAt(lastDevice_).transpose(embForHead);
    }
  }

  // The tied LM head holds the same weights as the embedding table, one
  // transpose apart, so gather the embedding straight out of it. embeddingCol
  // handles an fp16 or Q8 head; an affine-Q4 head is not supported, and a
  // pipeline split would put the head on the wrong device.
  const bool tiedHead = !reader.has_tensor("output.weight");
  const bool headUsableForEmbedding =
      output_weight_.isQ8() ||
      (!output_weight_.isQuantized() &&
       static_cast<bool>(output_weight_.handle));
  embedFromHead_ =
      tiedHead && headUsableForEmbedding && firstDevice_ == lastDevice_;

  if (!embedFromHead_) {
    // Separate table: fp16 [vocab, dim]. Float16 halves what is otherwise the
    // largest single allocation in most models.
    auto embd = reader.read_tensor_f32("token_embd.weight");
    std::vector<uint16_t> half(embd.size());
    for (size_t i = 0; i < embd.size(); ++i)
      half[i] = f32ToF16(embd[i]);
    embeddingTable_ = runtime_->createTensor(
        {config_.vocab_size, config_.dim}, cut::DataType::Float16, half.data(),
        false, firstDevice_);
  }

  // Initialize KV caches with pre-allocated GPU buffers.
  // Float16 KV cache halves memory (matches llama.cpp default) while
  // attention computation stays Float32 for numerical stability.
  kv_caches_.resize(config_.n_layers);
  for (uint32_t l = 0; l < config_.n_layers; ++l) {
    const size_t dev = layerDevice_[l];
    kv_caches_[l].k_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float16, false,
        dev);
    kv_caches_[l].v_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float16, false,
        dev);
  }

  // Pre-allocate per-device buffers for command buffer reuse.
  // Use mapped (host-visible) memory for small per-token params to avoid
  // staging command buffer + fence wait overhead on every token.
  runtimeParamsBuffers_.assign(runtime_->deviceCount(), {});
  hiddenBuffers_.assign(runtime_->deviceCount(), {});
  attnOutBuffers_.assign(runtime_->deviceCount(), {});
  embedOutBuffers_.assign(runtime_->deviceCount(), {});
  for (uint32_t s : segmentStart_) {
    const size_t dev = layerDevice_[s];
    if (runtimeParamsBuffers_[dev]) {
      continue; // already created for this device
    }
    runtimeParamsBuffers_[dev] =
        runtime_->createTensorMapped({2}, cut::DataType::UInt32, nullptr, dev);
    hiddenBuffers_[dev] = runtime_->createTensorEmpty(
        {config_.dim}, cut::DataType::Float32, false, dev);
    attnOutBuffers_[dev] = runtime_->createTensorEmpty(
        {config_.n_heads * config_.head_dim}, cut::DataType::Float32, false,
        dev);
    embedOutBuffers_[dev] = runtime_->createTensorEmpty(
        {config_.dim}, cut::DataType::Float16, false, dev);
  }
  tokenIdBuffer_ = runtime_->createTensorMapped({1}, cut::DataType::UInt32,
                                                nullptr, firstDevice_);
  // Initialize penalty factors to 1.0 (no penalty) for the first forward pass
  {
    std::vector<float> ones(config_.vocab_size, 1.0f);
    penaltyFactorsBuffer_ = runtime_->createTensorMapped(
        {1, config_.vocab_size}, cut::DataType::Float32, ones.data(),
        lastDevice_);
  }
  argmaxResultBuffer_ = runtime_->createTensorEmpty(
      {1}, cut::DataType::Float32, false, lastDevice_);

  // Precompute RoPE tables
  precomputeRoPE();

  // Build and optimize graph templates for forward pass
  auto graphStart = std::chrono::high_resolution_clock::now();
  buildGraphTemplates();
  auto graphEnd = std::chrono::high_resolution_clock::now();

  for (size_t d = 0; d < runtime_->deviceCount(); ++d) {
    runtime_->flush(d);
  }

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
  size_t totalBuffers = 0;
  double totalMB = 0.0;
  size_t totalBytes = 0;
  for (size_t d = 0; d < runtime_->deviceCount(); ++d) {
    totalBuffers += runtime_->bufferCount(d);
    totalBytes += runtime_->activeBufferMemoryBytes(d);
    totalMB += runtime_->activeBufferMemoryBytes(d) / (1024.0 * 1024.0);
  }
  std::cout << "Model loaded successfully. Buffers: " << totalBuffers
            << "  GPU memory: " << totalMB << " MB\n";

  if (memReportEnabled()) {
    size_t attnW = 0, ffnW = 0, norms = 0, embed = 0, lmHead = 0, kv = 0,
           rope = 0, misc = 0;
    for (size_t l = 0; l < layers_.size(); ++l) {
      const size_t d = layerDevice_[l];
      attnW += weightBytes(*runtime_, layers_[l].wq, d);
      attnW += weightBytes(*runtime_, layers_[l].wk, d);
      attnW += weightBytes(*runtime_, layers_[l].wv, d);
      attnW += weightBytes(*runtime_, layers_[l].wo, d);
      attnW += weightBytes(*runtime_, layers_[l].wqkv, d);
      ffnW += weightBytes(*runtime_, layers_[l].w_gate, d);
      ffnW += weightBytes(*runtime_, layers_[l].w_up, d);
      ffnW += weightBytes(*runtime_, layers_[l].w_down, d);
      ffnW += weightBytes(*runtime_, layers_[l].w_gate_up, d);
      norms += handleBytes(*runtime_, layers_[l].attn_norm, d);
      norms += handleBytes(*runtime_, layers_[l].ffn_norm, d);
      norms += handleBytes(*runtime_, layers_[l].bq, d);
      norms += handleBytes(*runtime_, layers_[l].bk, d);
      norms += handleBytes(*runtime_, layers_[l].bv, d);
      kv += handleBytes(*runtime_, kv_caches_[l].k_cache, d);
      kv += handleBytes(*runtime_, kv_caches_[l].v_cache, d);
    }
    norms += handleBytes(*runtime_, output_norm_, lastDevice_);
    embed = handleBytes(*runtime_, embeddingTable_, firstDevice_);
    lmHead = weightBytes(*runtime_, output_weight_, lastDevice_);
    misc += handleBytes(*runtime_, penaltyFactorsBuffer_, lastDevice_);
    misc += handleBytes(*runtime_, argmaxResultBuffer_, lastDevice_);
    misc += handleBytes(*runtime_, tokenIdBuffer_, firstDevice_);
    for (size_t d = 0; d < runtime_->deviceCount(); ++d) {
      if (d < ropeCosGpu_.size())
        rope += handleBytes(*runtime_, ropeCosGpu_[d], d);
      if (d < ropeSinGpu_.size())
        rope += handleBytes(*runtime_, ropeSinGpu_[d], d);
      if (d < runtimeParamsBuffers_.size())
        misc += handleBytes(*runtime_, runtimeParamsBuffers_[d], d);
      if (d < hiddenBuffers_.size())
        misc += handleBytes(*runtime_, hiddenBuffers_[d], d);
      if (d < attnOutBuffers_.size())
        misc += handleBytes(*runtime_, attnOutBuffers_[d], d);
      if (d < embedOutBuffers_.size())
        misc += handleBytes(*runtime_, embedOutBuffers_[d], d);
    }

    auto mb = [](size_t b) { return b / (1024.0 * 1024.0); };
    const size_t accounted =
        attnW + ffnW + norms + embed + lmHead + kv + rope + misc;
    std::cout << "  [mem] attn weights : " << mb(attnW) << " MB\n";
    std::cout << "  [mem] ffn weights  : " << mb(ffnW) << " MB\n";
    std::cout << "  [mem] norms+bias   : " << mb(norms) << " MB\n";
    std::cout << "  [mem] embedding    : " << mb(embed) << " MB\n";
    std::cout << "  [mem] lm head      : " << mb(lmHead) << " MB\n";
    std::cout << "  [mem] kv cache     : " << mb(kv) << " MB\n";
    std::cout << "  [mem] rope tables  : " << mb(rope) << " MB\n";
    std::cout << "  [mem] misc buffers : " << mb(misc) << " MB\n";
    std::cout << "  [mem] ---------------\n";
    std::cout << "  [mem] accounted    : " << mb(accounted) << " MB\n";
    std::cout << "  [mem] unaccounted  : " << mb(totalBytes - accounted)
              << " MB\n";
    std::cout << "  [mem] total        : " << mb(totalBytes) << " MB\n";
  }

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

  // Warmup: prefillBatched + forward + forwardPrefill at a representative N
  // so all three hot paths' Vulkan pipelines and transient buffer sizes are
  // pre-created. Costs ~80ms once at load; subsequent generate() avoids
  // the same ~40ms cold-start cost on both prefill (whichever path it uses)
  // and decode.
  {
    auto t_warmup_start = std::chrono::high_resolution_clock::now();
    const uint32_t warmupN = 16;
    std::vector<int> dummy(warmupN, bos_token_id_ >= 0 ? bos_token_id_ : 0);
    (void)prefillBatched(dummy);
    // Warm the decode path (records cached decode CB).
    (void)forward(dummy[0], static_cast<int>(warmupN));
    // Warm the per-token forwardPrefill path (records cached prefill CB).
    forwardPrefill(dummy[0], 0);
    resetCache();
    auto t_warmup_end = std::chrono::high_resolution_clock::now();
    std::cout << "  Warmup (N=" << warmupN << "): "
              << std::chrono::duration<double, std::milli>(t_warmup_end - t_warmup_start).count()
              << " ms\n";
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

  // Upload as 1D GPU tensors on every device that runs layers
  // (shader indexes linearly: pos * halfDim + i).
  ropeCosGpu_.assign(runtime_->deviceCount(), {});
  ropeSinGpu_.assign(runtime_->deviceCount(), {});
  for (uint32_t s : segmentStart_) {
    const size_t dev = layerDevice_[s];
    if (ropeCosGpu_[dev]) {
      continue; // already created for this device
    }
    ropeCosGpu_[dev] = runtime_->createTensor({config_.max_seq_len * half_dim},
                                              cut::DataType::Float32,
                                              cos_table.data(), false, dev);
    ropeSinGpu_[dev] = runtime_->createTensor({config_.max_seq_len * half_dim},
                                              cut::DataType::Float32,
                                              sin_table.data(), false, dev);
  }
}

// ============================================================================
// Graph template builders
// ============================================================================

GraphTemplate LlamaModel::buildQKVProjectionGraph(const LlamaLayer &layer,
                                                  size_t deviceId) {
  cut::graph::GraphBuilder builder(*runtime_, deviceId);
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

  if (layer.wqkv.isPresent() && !layer.bq) {
    // Fused QKV: single matmul → outputs [qdim + 2*kvdim] combined buffer.
    // The caller splits Q/K/V using createTensorView after graph execution.
    int32_t qdim = static_cast<int32_t>(config_.n_heads * config_.head_dim);
    int32_t kvdim = static_cast<int32_t>(config_.kv_dim);
    int32_t total = qdim + 2 * kvdim;

    // graphWeight dispatches to the quantized 3-input matmul when the
    // fused QKV weight is Q8_0.
    auto qkv_out = graphWeight(builder, layer.wqkv, normed_2d);
    auto qkv_flat = builder.ops().reshape(qkv_out, {total});

    builder.markOutput(qkv_flat);
    auto graph = builder.build();

    GraphTemplate tpl;
    tpl.dynamicInputIds = {graph->nodeId(vHidden)};
    tpl.preOptGraph = graph->clone();
    auto optimizer = cut::graph::GraphOptimizer::createDefault();
    optimizer.optimize(*graph, runtime_->store(deviceId));
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
  optimizer.optimize(*graph, runtime_->store(deviceId));
  tpl.stats = optimizer.stats();
  tpl.graph = std::move(graph);
  return tpl;
}

GraphTemplate
LlamaModel::buildAttnOutputResidualGraph(const LlamaLayer &layer,
                                         size_t deviceId) {
  cut::graph::GraphBuilder builder(*runtime_, deviceId);
  int32_t dim = static_cast<int32_t>(config_.dim);
  int32_t qdim = static_cast<int32_t>(config_.n_heads * config_.head_dim);

  // Dynamic inputs — each must use a DIFFERENT placeholder tensor so that
  // Operations can distinguish them during graph construction. The attention
  // output is [n_heads*head_dim], which differs from dim for models with an
  // explicit head size — use the per-device attn buffer as its placeholder.
  auto vAttnOut =
      builder.input(attnOutBuffers_[deviceId], /*isConstant=*/false);
  auto vHidden = builder.input(layer.ffn_norm, /*isConstant=*/false);

  auto attn_2d = builder.ops().reshape(vAttnOut, {1, qdim});
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
  optimizer.optimize(*graph, runtime_->store(deviceId));
  tpl.stats = optimizer.stats();
  tpl.graph = std::move(graph);
  return tpl;
}

GraphTemplate LlamaModel::buildFFNResidualGraph(const LlamaLayer &layer,
                                                size_t deviceId) {
  cut::graph::GraphBuilder builder(*runtime_, deviceId);
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
  if (layer.w_gate_up.isPresent()) {
    // Fused gate+up: single matmul → [1, 2*ffn_dim], then slice into gate/up.
    auto gate_up_out = graphWeight(builder, layer.w_gate_up, x_2d);
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
  optimizer.optimize(*graph, runtime_->store(deviceId));
  tpl.stats = optimizer.stats();
  tpl.graph = std::move(graph);
  return tpl;
}

GraphTemplate LlamaModel::buildLogitsGraph() {
  cut::graph::GraphBuilder builder(*runtime_, lastDevice_);
  int32_t dim = static_cast<int32_t>(config_.dim);

  // Dynamic input: hidden state [dim] — use the LAST layer's attn_norm as
  // shape placeholder: it lives on lastDevice_ like this graph, and differs
  // from output_norm_ used as constant input below.
  auto vHidden = builder.input(layers_[config_.n_layers - 1].attn_norm,
                               /*isConstant=*/false);

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
  optimizer.optimize(*graph, runtime_->store(lastDevice_));
  tpl.stats = optimizer.stats();
  tpl.graph = std::move(graph);
  return tpl;
}

void LlamaModel::buildGraphTemplates() {
  // One executor per device that runs graphs (layer devices + logits device).
  executors_.clear();
  executors_.resize(runtime_->deviceCount());
  for (uint32_t s : segmentStart_) {
    const size_t dev = layerDevice_[s];
    if (!executors_[dev]) {
      executors_[dev] = std::make_unique<cut::graph::GraphExecutor>(
          runtime_->ops(dev), runtime_->store(dev));
    }
  }
  if (!executors_[lastDevice_]) {
    executors_[lastDevice_] = std::make_unique<cut::graph::GraphExecutor>(
        runtime_->ops(lastDevice_), runtime_->store(lastDevice_));
  }

  // Collect optimization statistics across all graph templates
  std::map<std::string, int> totalOptimizations;

  // Build layer 0 graphs and collect stats (representative)
  if (config_.n_layers > 0) {
    layerGraphs_.resize(config_.n_layers);

    auto qkvTpl = buildQKVProjectionGraph(layers_[0], layerDevice_[0]);
    for (const auto &stat : qkvTpl.stats) {
      totalOptimizations[stat.name] += stat.runCount;
    }
    layerGraphs_[0].qkvProjection = std::move(qkvTpl);

    auto attnTpl = buildAttnOutputResidualGraph(layers_[0], layerDevice_[0]);
    for (const auto &stat : attnTpl.stats) {
      totalOptimizations[stat.name] += stat.runCount;
    }
    layerGraphs_[0].attnOutputResidual = std::move(attnTpl);

    auto ffnTpl = buildFFNResidualGraph(layers_[0], layerDevice_[0]);
    for (const auto &stat : ffnTpl.stats) {
      totalOptimizations[stat.name] += stat.runCount;
    }
    layerGraphs_[0].ffnResidual = std::move(ffnTpl);

    // Build remaining layers (reuse same patterns, multiply stats)
    for (uint32_t i = 1; i < config_.n_layers; ++i) {
      layerGraphs_[i].qkvProjection =
          buildQKVProjectionGraph(layers_[i], layerDevice_[i]);
      layerGraphs_[i].attnOutputResidual =
          buildAttnOutputResidualGraph(layers_[i], layerDevice_[i]);
      layerGraphs_[i].ffnResidual =
          buildFFNResidualGraph(layers_[i], layerDevice_[i]);
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

std::vector<cut::Tensor>
LlamaModel::executeGraph(GraphTemplate &tpl,
                         const std::vector<cut::ComputeHandle> &dynamicHandles,
                         size_t deviceId) {
  for (size_t i = 0; i < tpl.dynamicInputIds.size(); ++i) {
    auto &gn = tpl.graph->node(tpl.dynamicInputIds[i]);
    static_cast<cut::InputOpNode *>(gn.op.get())
        ->setGpuHandle(dynamicHandles[i]);
  }
  return executors_[deviceId]->execute(*tpl.graph);
}

void LlamaModel::splitQKV(const std::vector<cut::Tensor> &qkv,
                          cut::ComputeHandle &q,
                          cut::ComputeHandle &k,
                          cut::ComputeHandle &v,
                          size_t deviceId) {
  if (qkv.size() == 1) {
    // Fused QKV: split the combined [qdim + 2*kvdim] buffer into Q, K, V
    // views. Explicit barrier needed because the barrier tracker doesn't see
    // views as sharing the matmul's parent buffer.
    uint32_t qdim = config_.n_heads * config_.head_dim;
    uint32_t kvdim = config_.kv_dim;
    q = runtime_->store(deviceId).createTensorView(qkv[0], 0, {qdim},
                                                   cut::DataType::Float32);
    k = runtime_->store(deviceId).createTensorView(
        qkv[0], qdim * sizeof(float), {kvdim}, cut::DataType::Float32);
    v = runtime_->store(deviceId).createTensorView(
        qkv[0], (qdim + kvdim) * sizeof(float), {kvdim},
        cut::DataType::Float32);
    opsAt(deviceId).barrier();
  } else {
    q = qkv[0];
    k = qkv[1];
    v = qkv[2];
  }
}

cut::ComputeHandle LlamaModel::runLayer(uint32_t layerIdx,
                                        const cut::ComputeHandle &hidden) {
  auto &lg = layerGraphs_[layerIdx];
  const size_t dev = layerDevice_[layerIdx];
  auto &ops = opsAt(dev);

  auto qkv = executeGraph(lg.qkvProjection, {hidden}, dev);

  cut::ComputeHandle q, k, v;
  splitQKV(qkv, q, k, v, dev);

  auto &cache = kv_caches_[layerIdx];
  // RoPE(q/k) + K/V cache writes fused into one dispatch; attention stays a
  // separate dispatch so the cache writes are ordered by the inter-dispatch
  // barrier (the retired FusedAttention relied on cross-workgroup visibility
  // of cache writes, which is undefined behavior in Vulkan).
  // CUT_NO_FUSED_ROPE=1 restores the split 4-dispatch path for A/B tests.
  static const bool kNoFusedRope = std::getenv("CUT_NO_FUSED_ROPE") != nullptr;
  cut::Tensor q_roped;
  if (!kNoFusedRope) {
    q_roped = ops.fusedRoPEKVWrite(q, k, v, runtimeParamsBuffers_[dev],
                                   ropeCosGpu_[dev], ropeSinGpu_[dev],
                                   cache.k_cache, cache.v_cache,
                                   config_.head_dim);
    ops.barrier();
  } else {
    q_roped = ops.applyRoPE(q, ropeCosGpu_[dev], ropeSinGpu_[dev],
                            runtimeParamsBuffers_[dev], config_.head_dim);
    auto k_roped = ops.applyRoPE(k, ropeCosGpu_[dev], ropeSinGpu_[dev],
                                 runtimeParamsBuffers_[dev], config_.head_dim);
    ops.cacheWrite(cache.k_cache, k_roped, runtimeParamsBuffers_[dev]);
    ops.cacheWrite(cache.v_cache, v, runtimeParamsBuffers_[dev]);
  }
  ops.attention(q_roped, cache.k_cache, cache.v_cache,
                runtimeParamsBuffers_[dev], config_.n_heads,
                config_.n_kv_heads, config_.head_dim, attnOutBuffers_[dev]);

  auto attn_result =
      executeGraph(lg.attnOutputResidual, {attnOutBuffers_[dev], hidden}, dev);
  auto ffn_result = executeGraph(lg.ffnResidual, {attn_result[0]}, dev);
  return ffn_result[0];
}

// ============================================================================
// Forward pass
// ============================================================================

cut::ComputeHandle LlamaModel::forward(int token_id, int pos) {
  uint32_t upos = static_cast<uint32_t>(pos);

  // Update runtime params {pos, seqLen} on every device that runs layers
  // (mapped writes; duplicate writes to a device are harmless memcpys).
  uint32_t params[2] = {upos, upos + 1};
  for (uint32_t s : segmentStart_) {
    const size_t dev = layerDevice_[s];
    runtime_->copyToTensor(runtimeParamsBuffers_[dev], params, sizeof(params),
                           0, 0, dev);
  }

  // Update token ID for GPU embedding lookup (first device).
  uint32_t tid = static_cast<uint32_t>(token_id);
  runtime_->copyToTensor(tokenIdBuffer_, &tid, sizeof(uint32_t), 0, 0,
                         firstDevice_);

  if (decodeCBCached_) {
    // Resubmit each segment's cached CB in pipeline order, copying the
    // hidden state across device boundaries between segments. Mapped
    // buffers (runtimeParams, tokenId, penaltyFactors) were updated via
    // direct memcpy above — no staging fence needed.
    for (size_t seg = 0; seg < cachedDecodeCBs_.size(); ++seg) {
      const size_t dev = layerDevice_[segmentStart_[seg]];
      runtime_->resubmitAndWait(cachedDecodeCBs_[seg], dev);
      if (seg + 1 < cachedDecodeCBs_.size()) {
        const size_t nextDev = layerDevice_[segmentStart_[seg + 1]];
        runtime_->transferTensor(segmentDecodeOut_[seg], dev,
                                 hiddenBuffers_[nextDev], nextDev);
      }
    }
    if (dispatchProfileEnabled())
      dumpDispatchProfile(*runtime_, "decode");
    return argmaxResultBuffer_;
  }

  // --- First forward: record one reusable CB per pipeline segment ---
  cachedDecodeCBs_.clear();
  segmentDecodeOut_.clear();

  cut::ComputeHandle hidden =
      embedTokens(tokenIdBuffer_, hiddenBuffers_[firstDevice_]);
  bool allCached = true;
  for (uint32_t i = 0; i < config_.n_layers; ++i) {
    hidden = runLayer(i, hidden);

    const bool lastLayer = (i + 1 == config_.n_layers);
    const bool segmentEnd = lastLayer || isSegmentStart(i + 1);
    if (!segmentEnd) {
      continue;
    }
    const size_t dev = layerDevice_[i];
    if (lastLayer) {
      // LM head logits + penalty + argmax live in the last segment's CB so
      // sampling needs no extra submit. penaltyFactorsBuffer_ is updated via
      // memcpy before each resubmit.
      auto logit_result = executeGraph(logitsGraph_, {hidden}, dev);
      logitsOutput_ = logit_result[0];
      auto penalized =
          opsAt(dev).repetitionPenalty(logitsOutput_, penaltyFactorsBuffer_);
      argmaxResultBuffer_ = opsAt(dev).reduce(cut::ReduceArgmax, penalized);
    }
    segmentDecodeOut_.push_back(hidden);
    cachedDecodeCBs_.push_back(runtime_->submitReusable(dev));
    allCached = allCached && static_cast<bool>(cachedDecodeCBs_.back());
    if (!lastLayer) {
      const size_t nextDev = layerDevice_[i + 1];
      runtime_->transferTensor(hidden, dev, hiddenBuffers_[nextDev], nextDev);
      hidden = hiddenBuffers_[nextDev];
    }
  }
  decodeCBCached_ = allCached && !cachedDecodeCBs_.empty();

  return argmaxResultBuffer_;
}

int LlamaModel::decodeStep(int token_id, int pos) {
  auto argmaxBuf = forward(token_id, pos);
  float best = 0.0f;
  runtime_->copyFromTensor(argmaxBuf, &best, sizeof(float), 0, 0, lastDevice_);
  return static_cast<int>(best);
}

void LlamaModel::setStopTokensSuppressed(bool suppressed) {
  std::vector<float> factors(config_.vocab_size, 1.0f);
  if (suppressed) {
    if (eos_token_id_ >= 0 &&
        static_cast<uint32_t>(eos_token_id_) < config_.vocab_size) {
      factors[eos_token_id_] = 1e9f;
    }
    for (int st : stopTokenIds_) {
      if (st >= 0 && static_cast<uint32_t>(st) < config_.vocab_size) {
        factors[st] = 1e9f;
      }
    }
  }
  runtime_->copyToTensor(penaltyFactorsBuffer_, factors.data(),
                         config_.vocab_size * sizeof(float), 0, 0,
                         lastDevice_);
}

void LlamaModel::forwardPrefill(int token_id, int pos) {
  uint32_t upos = static_cast<uint32_t>(pos);
  uint32_t params[2] = {upos, upos + 1};
  for (uint32_t s : segmentStart_) {
    const size_t dev = layerDevice_[s];
    runtime_->copyToTensor(runtimeParamsBuffers_[dev], params, sizeof(params),
                           0, 0, dev);
  }

  uint32_t tid = static_cast<uint32_t>(token_id);
  runtime_->copyToTensor(tokenIdBuffer_, &tid, sizeof(uint32_t), 0, 0,
                         firstDevice_);

  if (prefillCBCached_) {
    for (size_t seg = 0; seg < cachedPrefillCBs_.size(); ++seg) {
      const size_t dev = layerDevice_[segmentStart_[seg]];
      runtime_->resubmitAndWait(cachedPrefillCBs_[seg], dev);
      if (seg + 1 < cachedPrefillCBs_.size()) {
        const size_t nextDev = layerDevice_[segmentStart_[seg + 1]];
        runtime_->transferTensor(segmentPrefillOut_[seg], dev,
                                 hiddenBuffers_[nextDev], nextDev);
      }
    }
    return;
  }

  // Record per-segment prefill CBs: embedding → layers (no logits/argmax).
  cachedPrefillCBs_.clear();
  segmentPrefillOut_.clear();

  cut::ComputeHandle hidden =
      embedTokens(tokenIdBuffer_, hiddenBuffers_[firstDevice_]);
  bool allCached = true;
  for (uint32_t i = 0; i < config_.n_layers; ++i) {
    hidden = runLayer(i, hidden);
    const bool lastLayer = (i + 1 == config_.n_layers);
    const bool segmentEnd = lastLayer || isSegmentStart(i + 1);
    if (!segmentEnd) {
      continue;
    }
    const size_t dev = layerDevice_[i];
    segmentPrefillOut_.push_back(hidden);
    cachedPrefillCBs_.push_back(runtime_->submitReusable(dev));
    allCached = allCached && static_cast<bool>(cachedPrefillCBs_.back());
    if (!lastLayer) {
      const size_t nextDev = layerDevice_[i + 1];
      runtime_->transferTensor(hidden, dev, hiddenBuffers_[nextDev], nextDev);
      hidden = hiddenBuffers_[nextDev];
    }
  }
  prefillCBCached_ = allCached && !cachedPrefillCBs_.empty();
}

int LlamaModel::prefill(const std::vector<int> &tokens) {
  if (segmentStart_.size() > 1) {
    // Pipeline split: the single-CB inline-update recording below is
    // single-device only. Use the per-token cached-CB path instead.
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
      forwardPrefill(tokens[i], static_cast<int>(i));
    }
    auto argmaxBuf =
        forward(tokens.back(), static_cast<int>(tokens.size() - 1));
    float best = 0.0f;
    runtime_->copyFromTensor(argmaxBuf, &best, sizeof(float), 0, 0,
                             lastDevice_);
    return static_cast<int>(best);
  }

  // Record all prompt tokens into a single command buffer using inline buffer
  // updates (vkCmdUpdateBuffer) for runtimeParams and tokenId. This eliminates
  // N-1 fence waits by batching everything into one CB submission.
  for (size_t i = 0; i < tokens.size(); ++i) {
    uint32_t upos = static_cast<uint32_t>(i);
    uint32_t params[2] = {upos, upos + 1};
    runtime_->updateBufferInline(runtimeParamsBuffers_[firstDevice_], params,
                                 sizeof(params), firstDevice_);

    uint32_t tid = static_cast<uint32_t>(tokens[i]);
    runtime_->updateBufferInline(tokenIdBuffer_, &tid, sizeof(uint32_t),
                                 firstDevice_);

    auto hidden = embedTokens(tokenIdBuffer_, hiddenBuffers_[firstDevice_]);
    for (uint32_t l = 0; l < config_.n_layers; ++l)
      hidden = runLayer(l, hidden);

    // Only compute logits + argmax for the last token
    if (i == tokens.size() - 1) {
      auto logit_result = executeGraph(logitsGraph_, {hidden}, lastDevice_);
      logitsOutput_ = logit_result[0];

      auto penalized = opsAt(lastDevice_)
                           .repetitionPenalty(logitsOutput_,
                                              penaltyFactorsBuffer_);
      argmaxResultBuffer_ =
          opsAt(lastDevice_).reduce(cut::ReduceArgmax, penalized);
    }
  }

  // Single submit + wait for the entire prefill
  runtime_->flushPendingCommands(firstDevice_);
  if (dispatchProfileEnabled())
    dumpDispatchProfile(*runtime_, "prefill-per-token");

  // Read back the argmax result (4 bytes)
  float best = 0.0f;
  runtime_->copyFromTensor(argmaxResultBuffer_, &best, sizeof(float), 0, 0,
                           lastDevice_);
  return static_cast<int>(best);
}

int LlamaModel::prefillBatched(const std::vector<int> &tokens) {
  uint32_t N = static_cast<uint32_t>(tokens.size());
  uint32_t dim = config_.dim;
  uint32_t qdim = config_.n_heads * config_.head_dim;
  uint32_t kvdim = config_.kv_dim;
  uint32_t alignedKvDim = (kvdim + 3) & ~3u;

  static const bool kTiming = std::getenv("CUT_PREFILL_TIMING") != nullptr;
  auto tStart = std::chrono::high_resolution_clock::now();

  // 1. Upload all token IDs [N] and position array [0..N-1].
  //    Positions are needed by the batched attention ops on every device
  //    that runs layers, so replicate the buffer per device.
  std::vector<uint32_t> tokenIds(tokens.begin(), tokens.end());
  auto tokenBuf = runtime_->createTensor({N}, cut::DataType::UInt32,
                                         tokenIds.data(), false, firstDevice_);

  std::vector<uint32_t> positions(N);
  for (uint32_t i = 0; i < N; ++i)
    positions[i] = i;
  std::vector<cut::ComputeHandle> posBufs(runtime_->deviceCount());
  for (uint32_t s : segmentStart_) {
    const size_t d = layerDevice_[s];
    if (!posBufs[d]) {
      posBufs[d] = runtime_->createTensor({N}, cut::DataType::UInt32,
                                          positions.data(), false, d);
    }
  }

  auto tAfterInputs = std::chrono::high_resolution_clock::now();

  // 2. Embedding: [N] → [N, dim]. The table is Float16, so cast the lookup
  //    result up to the Float32 the rest of the pass expects.
  auto hidden = embedTokens(tokenBuf);

  // 3. Process all layers
  for (uint32_t l = 0; l < config_.n_layers; ++l) {
    auto &layer = layers_[l];
    auto &cache = kv_caches_[l];
    const size_t dev = layerDevice_[l];
    auto &ops = opsAt(dev);

    // Cross-device boundary: move the [N, dim] activations to this layer's
    // device through the host.
    if (l > 0 && dev != layerDevice_[l - 1]) {
      hidden = runtime_->transferTensor(hidden, layerDevice_[l - 1], dev);
    }

    // --- QKV projection: RMSNorm → batched matmul ---
    auto normed = ops.rmsNorm(hidden, layer.attn_norm, config_.norm_eps);

    // Use the SAME fused weight as the decode path for numerical equivalence.
    // Fused: one matmul → [N, qdim+2*kvdim], views index Q/K/V per-row.
    // Separate: 3 matmuls → [N, qdim], [N, kvdim], [N, kvdim].
    uint32_t total = qdim + 2 * kvdim;
    uint32_t alignedTotal = (total + 3) & ~3u;
    uint32_t alignedQdim = (qdim + 3) & ~3u;
    bool fusedQKV = layer.wqkv.isPresent() && !layer.bq;

    cut::ComputeHandle qBuf, kBuf, vBuf;
    uint32_t qRowStride, kRowStride, vRowStride;
    uint32_t qColOff, kColOff, vColOff;

    if (fusedQKV) {
      auto qkvOut = layer.wqkv.isQuantized()
                        ? ops.matmul(normed, layer.wqkv.qValues,
                                     layer.wqkv.qScales)
                        : ops.matmul(normed, layer.wqkv.handle);
      qBuf = qkvOut;
      kBuf = qkvOut;
      vBuf = qkvOut;
      qRowStride = kRowStride = vRowStride = alignedTotal;
      qColOff = 0;
      kColOff = qdim;
      vColOff = qdim + kvdim;
    } else {
      qBuf = layer.wq.isQuantized()
                 ? ops.matmul(normed, layer.wq.qValues, layer.wq.qScales)
                 : ops.matmul(normed, layer.wq.handle);
      kBuf = layer.wk.isQuantized()
                 ? ops.matmul(normed, layer.wk.qValues, layer.wk.qScales)
                 : ops.matmul(normed, layer.wk.handle);
      vBuf = layer.wv.isQuantized()
                 ? ops.matmul(normed, layer.wv.qValues, layer.wv.qScales)
                 : ops.matmul(normed, layer.wv.handle);
      // Row-broadcast: prefill produces [N, qdim]/[N, kvdim] activations while
      // the bias is a single [qdim]/[kvdim] row, so it must be added to every
      // one of the N rows. Plain binaryOp requires equal element counts and
      // throws for N > 1 (decode is fine only because there N == 1).
      if (layer.bq)
        qBuf = ops.binaryOpRowBcast(cut::BinaryAdd, qBuf, layer.bq);
      if (layer.bk)
        kBuf = ops.binaryOpRowBcast(cut::BinaryAdd, kBuf, layer.bk);
      if (layer.bv)
        vBuf = ops.binaryOpRowBcast(cut::BinaryAdd, vBuf, layer.bv);
      qRowStride = alignedQdim;
      kRowStride = vRowStride = alignedKvDim;
      qColOff = kColOff = vColOff = 0;
    }

    // Barrier so the attention path's reads of qBuf/kBuf/vBuf see the
    // just-written matmul outputs. The matmul's output handle differs from
    // any per-row view's handle, so the barrier tracker can't auto-insert
    // this for the strided-input batched ops below.
    ops.barrier();

    // --- Two-dispatch batched attention path ---
    // Replaces the per-token loop (5N dispatches/layer) with 2 dispatches:
    // 1. BatchedKVCacheWrite: applies RoPE to K and writes K/V to cache.
    // 2. BatchedAttentionReadCache: applies RoPE to Q and computes attention
    //    over the now-populated cache.
    // The Vulkan auto-barrier between dispatches (both touch kCache/vCache)
    // ensures D1's writes are visible to D2's reads — fixing the
    // cross-workgroup race in the old single-dispatch BatchedFusedAttention.
    ops.batchedKVCacheWrite(kBuf, vBuf, cache.k_cache, cache.v_cache,
                            posBufs[dev], ropeCosGpu_[dev], ropeSinGpu_[dev],
                            /*batchSize=*/ N,
                            /*nKvHeads=*/ config_.n_kv_heads,
                            /*headDim=*/ config_.head_dim,
                            /*kStride=*/ kRowStride,
                            /*vStride=*/ vRowStride,
                            /*kOffset=*/ kColOff,
                            /*vOffset=*/ vColOff);
    auto attnOut = ops.batchedAttentionReadCache(
        qBuf, cache.k_cache, cache.v_cache, posBufs[dev],
        ropeCosGpu_[dev], ropeSinGpu_[dev],
        /*batchSize=*/ N,
        /*nHeads=*/ config_.n_heads,
        /*nKvHeads=*/ config_.n_kv_heads,
        /*headDim=*/ config_.head_dim,
        /*qStride=*/ qRowStride,
        /*qOffset=*/ qColOff);

    // Barrier so the output projection reads the views' writes into attnOut.
    ops.barrier();

    // --- Output projection + residual ---
    // Tried matmulBinary fusion (would save the BinaryAdd dispatch), but
    // the extra dataD read in the matmul shader cost more wall-clock than
    // it saved. See git log for the experiment.
    cut::ComputeHandle proj;
    if (layer.wo.isQuantized()) {
      proj = ops.matmul(attnOut, layer.wo.qValues, layer.wo.qScales);
    } else {
      proj = ops.matmul(attnOut, layer.wo.handle);
    }
    hidden = ops.binaryOp(cut::BinaryAdd, proj, hidden);

    // --- FFN ---
    auto ffnNormed = ops.rmsNorm(hidden, layer.ffn_norm, config_.norm_eps);

    // FFN: 2 separate gate/up matmuls (gate fuses SiLU). Tried fusing the
    // up*gate multiply into the up matmul via matmulBinary, but that
    // regressed wall-clock — likely because the per-element fusion read of
    // gateOut at every matmul output cell hurts the FFN-sized matmul's
    // memory traffic more than it saves a dispatch.
    auto gateOut =
        layer.w_gate.isQuantized()
            ? ops.matmulUnary(cut::UnarySilu, ffnNormed, layer.w_gate.qValues,
                              layer.w_gate.qScales)
            : ops.matmulUnary(cut::UnarySilu, ffnNormed, layer.w_gate.handle);
    auto upOut =
        layer.w_up.isQuantized()
            ? ops.matmul(ffnNormed, layer.w_up.qValues, layer.w_up.qScales)
            : ops.matmul(ffnNormed, layer.w_up.handle);
    auto gateUpResult = ops.binaryOp(cut::BinaryMul, gateOut, upOut);

    // Down projection + residual
    cut::ComputeHandle down;
    if (layer.w_down.isQuantized()) {
      down = ops.matmul(gateUpResult, layer.w_down.qValues,
                        layer.w_down.qScales);
    } else {
      down = ops.matmul(gateUpResult, layer.w_down.handle);
    }
    hidden = ops.binaryOp(cut::BinaryAdd, down, hidden);
  }

  auto tAfterLayers = std::chrono::high_resolution_clock::now();

  // 4. Logits on last row only
  //    Extract row N-1 from [N, dim] → [dim]
  uint32_t alignedDim = (dim + 3) & ~3u;
  auto lastRow = runtime_->store(lastDevice_).createTensorView(
      hidden, static_cast<size_t>(N - 1) * alignedDim * sizeof(float), {dim},
      cut::DataType::Float32);

  // The view shares the parent's VkBuffer but has a different ComputeHandle,
  // so the barrier tracker doesn't see it as aliasing prior writes to
  // `hidden`. Without this, the logits rmsNorm can read stale data.
  opsAt(lastDevice_).barrier();

  // RMSNorm + logits matmul (reuse logits graph for single-row)
  auto logit_result = executeGraph(logitsGraph_, {lastRow}, lastDevice_);
  logitsOutput_ = logit_result[0];

  auto penalized =
      opsAt(lastDevice_).repetitionPenalty(logitsOutput_, penaltyFactorsBuffer_);
  argmaxResultBuffer_ = opsAt(lastDevice_).reduce(cut::ReduceArgmax, penalized);

  auto tAfterLogits = std::chrono::high_resolution_clock::now();

  // Single submit + wait
  runtime_->flushPendingCommands(lastDevice_);

  auto tAfterFlush = std::chrono::high_resolution_clock::now();

  if (dispatchProfileEnabled())
    dumpDispatchProfile(*runtime_, "prefill");

  float best = 0.0f;
  runtime_->copyFromTensor(argmaxResultBuffer_, &best, sizeof(float), 0, 0,
                           lastDevice_);

  auto tAfterRead = std::chrono::high_resolution_clock::now();

  if (kTiming) {
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    fprintf(stderr,
            "[prefill-timing] N=%u inputs=%.3fms layers=%.3fms logits=%.3fms "
            "flush=%.3fms readback=%.3fms total=%.3fms\n",
            N, ms(tStart, tAfterInputs), ms(tAfterInputs, tAfterLayers),
            ms(tAfterLayers, tAfterLogits), ms(tAfterLogits, tAfterFlush),
            ms(tAfterFlush, tAfterRead), ms(tStart, tAfterRead));
  }

  return static_cast<int>(best);
}

// ============================================================================
// Generation
// ============================================================================

GenerationResult LlamaModel::generate(const std::vector<int> &prompt_tokens,
                                      int max_new_tokens,
                                      float repeat_penalty,
                                      int repeat_last_n,
                                      int forceMinNewTokens) {
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
  // Suppress EOS/stop tokens during early generation to counteract
  // accumulated GPU FP32 precision drift through residual layers, which
  // can artificially elevate the EOS logit for longer prompts.
  int minNewTokens =
      forceMinNewTokens >= 0
          ? forceMinNewTokens
          : std::max(1, static_cast<int>(prompt_tokens.size()) / 4);
  int generatedCount = 0;

  auto uploadPenaltyFactors = [&](bool suppressEos = false) {
    if (!hasPenalty && !suppressEos)
      return;
    std::vector<float> factors(vocabSize, 1.0f);
    if (hasPenalty) {
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
    }
    if (suppressEos) {
      if (eos_token_id_ >= 0 &&
          static_cast<uint32_t>(eos_token_id_) < vocabSize)
        factors[eos_token_id_] = 1e9f;
      for (int st : stopTokenIds_) {
        if (st >= 0 && static_cast<uint32_t>(st) < vocabSize)
          factors[st] = 1e9f;
      }
    }
    runtime_->copyToTensor(penaltyFactorsBuffer_, factors.data(),
                           vocabSize * sizeof(float), 0, 0, lastDevice_);
  };

  // Read argmax result from GPU (4 bytes) after forward completes.
  auto readArgmax = [&](const cut::ComputeHandle &argmaxBuf) -> int {
    float best = 0.0f;
    runtime_->copyFromTensor(argmaxBuf, &best, sizeof(float), 0, 0,
                             lastDevice_);
    return static_cast<int>(best);
  };

  // Batched prefill: all N prompt tokens in one CB submission. The model
  // load step ran a warmup prefillBatched so pipelines and transient
  // buffer sizes are already cached, making this path faster than the
  // per-token forwardPrefill loop on cold-start.
  auto prefillStart = std::chrono::high_resolution_clock::now();
  uploadPenaltyFactors(generatedCount < minNewTokens);
  // CUT_PREFILL=per_token routes prefill through the per-token decode path —
  // a correctness fallback while the batched prefill ops have issues at
  // some model geometries (e.g. head_dim 128 / large K).
  const char *prefillMode = std::getenv("CUT_PREFILL");
  if (prefillMode && std::string(prefillMode) == "per_token") {
    next_token = prefill(prompt_tokens);
  } else {
    next_token = prefillBatched(prompt_tokens);
  }
  generatedCount++;

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

  // Stop on EOS from first sampled token (only if past min_new_tokens)
  if (isStopToken(next_token) && generatedCount >= minNewTokens) {
    result.tokens = std::move(tokens);
    result.generatedTokens = generatedCount;
    result.generateMs = 0.0;
    return result;
  }

  // Autoregressive generation.
  // CUT_BENCH_SYNTHETIC makes decode measure pure forward-pass throughput to
  // match llama.cpp's llama-bench tg test: skip the per-token penalty upload,
  // the argmax readback, per-token printing, and the early-stop check — feed a
  // fixed token and run the full length. forward() (the real per-token model
  // work) is unchanged.
  static const bool kBenchSynthetic = std::getenv("CUT_BENCH_SYNTHETIC") != nullptr;
  auto genStart = std::chrono::high_resolution_clock::now();
  for (int step = 0; step < max_new_tokens - 1; ++step) {
    int pos = static_cast<int>(prompt_tokens.size()) + step;

    if (!kBenchSynthetic) {
      // Upload penalty factors before forward (staged, flushed by resubmit)
      uploadPenaltyFactors(generatedCount < minNewTokens);
    }

    auto argmaxBuf = forward(next_token, pos);

    if (!kBenchSynthetic) {
      next_token = readArgmax(argmaxBuf);
    }
    (void)argmaxBuf;

    generatedCount++;
    tokens.push_back(next_token);

    if (!kBenchSynthetic) {
      std::cout << "Generated token: " << next_token << "\n";
      if (isStopToken(next_token) && generatedCount >= minNewTokens) {
        break;
      }
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
  runtime_->setProfilingEnabled(enabled);
}

void LlamaModel::resetCache() {
  for (uint32_t l = 0; l < config_.n_layers; ++l) {
    const size_t dev = layerDevice_[l];
    // Re-allocate fresh GPU cache buffers (Float16 to match initial allocation)
    kv_caches_[l].k_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float16, false,
        dev);
    kv_caches_[l].v_cache = runtime_->createTensorEmpty(
        {config_.max_seq_len, config_.kv_dim}, cut::DataType::Float16, false,
        dev);
    kv_caches_[l].seq_len = 0;
  }
  // Invalidate cached command buffers (new generation = new KV cache handles)
  cachedDecodeCBs_.clear();
  segmentDecodeOut_.clear();
  decodeCBCached_ = false;
  cachedPrefillCBs_.clear();
  segmentPrefillOut_.clear();
  prefillCBCached_ = false;
}

} // namespace gguf
