#pragma once

#include "Runtime.h"
#include "safetensor_reader.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ltx {

/// Model hyperparameters for LTX-Video 2B (diffusers layout). Fixed for MVP.
struct LtxConfig {
  uint32_t inChannels = 128;    // packed latent channels
  uint32_t dim = 2048;          // inner dim = heads * headDim
  uint32_t nHeads = 32;
  uint32_t headDim = 64;
  uint32_t nLayers = 28;
  uint32_t ffnDim = 8192;
  uint32_t captionDim = 4096;   // T5-XXL embedding width
  uint32_t timeFreqDim = 256;   // sinusoidal timestep embedding width
  float blockNormEps = 1e-6f;   // RMSNorm in blocks / final LayerNorm
  float qkNormEps = 1e-5f;      // attention QK RMSNorm
  float ropeTheta = 10000.0f;
  uint32_t ropeBaseFrames = 20, ropeBaseHeight = 2048, ropeBaseWidth = 2048;
  // FlowMatchEulerDiscreteScheduler config (scheduler_config.json)
  float baseShift = 0.95f, maxShift = 2.05f;
  uint32_t baseImageSeqLen = 1024, maxImageSeqLen = 4096;
  float shiftTerminal = 0.1f;
};

/// Attention weight set (self- or cross-attention) uploaded to the GPU.
struct AttnWeights {
  cut::ComputeHandle wq, wk, wv;   // Float16 [in, out] (already transposed for matmul)
  cut::ComputeHandle wo;           // Float16 [dim, dim] — consumed via per-head row views
  cut::ComputeHandle bq, bk, bv, bo; // Float32 [out]
  cut::ComputeHandle normQ, normK; // Float32 [dim]; normQ is PRE-SCALED by 1/sqrt(headDim)
};

/// One transformer block's weights.
struct BlockWeights {
  AttnWeights self_, cross_;
  cut::ComputeHandle ffnW1, ffnW2;   // Float16 [dim, ffnDim], [ffnDim, dim]
  cut::ComputeHandle ffnB1, ffnB2;   // Float32
  std::vector<float> scaleShiftTable; // CPU copy [6 * dim]
};

/// Read-only view over a set of safetensors shards; finds tensors by name in
/// whichever shard holds them.
class ShardedSafeTensors {
public:
  explicit ShardedSafeTensors(const std::vector<std::string> &paths);
  bool has(const std::string &name) const;
  std::vector<float> readF32(const std::string &name) const;
  std::vector<size_t> shape(const std::string &name) const;
private:
  std::vector<std::unique_ptr<safetensor::SafeTensorReader>> readers_;
};

/// LTX-Video 2B text-to-video DiT runner (MVP).
/// Text embeddings are precomputed offline; the returned latents are decoded
/// to video offline (see scripts/ltx_encode_prompt.py / ltx_decode_latents.py).
class LtxModel {
public:
  /// Load transformer weights from `<modelDir>/transformer/*.safetensors`.
  void load(const std::string &modelDir, cut::Runtime &runtime);

  const LtxConfig &config() const { return config_; }

  /// Run the full denoising loop.
  /// promptEmbeds / negativeEmbeds: row-major [nTokens, captionDim] Float32.
  /// latentFrames/Height/Width: latent-space dims (video dims / VAE compression).
  /// frameRate: source video frame rate (e.g. 25.0), used for RoPE scaling.
  /// Returns packed latents [S, inChannels], S = latentFrames*latentHeight*latentWidth.
  std::vector<float> generate(const std::vector<float> &promptEmbeds,
                              uint32_t promptTokens,
                              const std::vector<float> &negativeEmbeds,
                              uint32_t negativeTokens,
                              uint32_t latentFrames,
                              uint32_t latentHeight,
                              uint32_t latentWidth,
                              float frameRate,
                              uint32_t steps,
                              float guidanceScale,
                              uint32_t seed);

private:
  // ---- load helpers ----
  /// Upload a PyTorch Linear weight [out, in] as a transposed Float16 GPU
  /// tensor [in, out] ready for ops().matmul.
  cut::ComputeHandle uploadLinearWeightF16(const ShardedSafeTensors &st,
                                    const std::string &name);
  /// Upload a 1-D Float32 tensor (bias / norm weight), optionally scaling
  /// every element by `scale`.
  cut::ComputeHandle uploadVecF32(const ShardedSafeTensors &st,
                           const std::string &name,
                           float scale = 1.0f);
  AttnWeights loadAttn(const ShardedSafeTensors &st, const std::string &prefix);

  // ---- CPU-side helpers (documented in ltx.cpp) ----
  /// Sinusoidal timestep embedding + the two time_embed linears + final
  /// AdaLN-single linear, all in fp32 on the CPU. Fills:
  ///   outMod:  [nLayers * 6 * dim]  per-block modulation = block table + temb,
  ///            with the two `scale` rows already incremented by 1
  ///   outFinal:[2 * dim] final shift/scale (scale already incremented by 1)
  void computeTimestepModulation(float timestep,
                                 std::vector<float> &outMod,
                                 std::vector<float> &outFinal) const;
  /// LTX fractional 3D RoPE cos/sin tables for S = f*h*w tokens: [S, dim].
  void computeRopeTables(uint32_t latentFrames, uint32_t latentHeight,
                         uint32_t latentWidth, float frameRate,
                         std::vector<float> &cosTbl,
                         std::vector<float> &sinTbl) const;
  /// FlowMatch sigma schedule with resolution-dependent dynamic shifting and
  /// terminal stretch; returns `steps + 1` values (last is 0).
  std::vector<float> computeSigmas(uint32_t steps, uint32_t videoSeqLen) const;

  // ---- GPU forward helpers ----
  /// Multi-head bidirectional attention (composed from matmul/softmax ops).
  /// qSrc [Sq, dim], kvSrc [Skv, dim]; applies QK RMSNorm and (optionally)
  /// interleaved RoPE from ropeCos_/ropeSin_ (self-attention only).
  cut::ComputeHandle mha(const AttnWeights &w, const cut::ComputeHandle &qSrc,
                  const cut::ComputeHandle &kvSrc, uint32_t sq, uint32_t skv,
                  bool useRope);
  /// One transformer block. `modOffset` = block index * 6 * dim floats into
  /// the per-step modulation buffer modBuf_.
  cut::ComputeHandle block(const BlockWeights &bw, const cut::ComputeHandle &hidden,
                    const cut::ComputeHandle &encoder, uint32_t sVideo,
                    uint32_t sText, uint32_t blockIdx);
  /// Full DiT forward for one (latents, encoder) pair at the current step's
  /// modulation. Returns [S, inChannels] noise prediction.
  cut::ComputeHandle forward(const cut::ComputeHandle &latents, const cut::ComputeHandle &encoder,
                      uint32_t sVideo, uint32_t sText);

  LtxConfig config_;
  cut::Runtime *runtime_ = nullptr;

  // Top-level GPU weights
  cut::ComputeHandle projInW_, projInB_;    // [128->2048]
  cut::ComputeHandle projOutW_, projOutB_;  // [2048->128]
  cut::ComputeHandle capW1_, capB1_, capW2_, capB2_; // caption projection
  cut::ComputeHandle ones_;                 // Float32 [dim] of 1.0 (no-affine RMSNorm)
  std::vector<BlockWeights> blocks_;

  // CPU weights for the timestep path
  std::vector<float> tsL1W_, tsL1B_; // [2048,256], [2048]
  std::vector<float> tsL2W_, tsL2B_; // [2048,2048], [2048]
  std::vector<float> adaW_, adaB_;   // [6*2048, 2048], [6*2048]
  std::vector<float> finalScaleShiftTable_; // [2*2048]

  // Per-generation GPU state
  cut::ComputeHandle ropeCos_, ropeSin_;    // [S, dim]
  cut::ComputeHandle modBuf_;               // Float32 [nLayers * 6 * dim], per step
  cut::ComputeHandle finalModBuf_;          // Float32 [2 * dim], per step

  // Transient GPU tensors created since the last releaseTransients() call.
  // Every intermediate op output is tracked and bulk-released once the GPU
  // work consuming it has been flushed (per transformer block) — without
  // this, one forward pass at video resolutions exhausts VRAM.
  std::vector<cut::ComputeHandle> transients_;
  /// Track an op output for later release; returns the handle unchanged.
  cut::ComputeHandle track(cut::ComputeHandle h) {
    transients_.push_back(h);
    return h;
  }
  /// Flush pending GPU work, then drop the tracked references so the
  /// refcounted buffers get recycled.
  void releaseTransients();
};

} // namespace ltx
