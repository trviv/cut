#include "MatMulOp.h"
#include "Shaders.h"
#include <cstdlib>
#include "TensorStore.h"
#include "VariantSelector.h"

namespace cut {

// Cooperative matrix variant indices (appended after the standard variants)
static constexpr int kCoopMatVariant = kMatMulVariantCount - 3;
static constexpr int kCoopMatTiledVariant = kMatMulVariantCount - 2;
static constexpr int kCoopMatGemvVariant = kMatMulVariantCount - 1;

// Helper: check if cooperative matrix variant should be auto-selected
static bool shouldUseCoopMat(const DeviceCaps &caps,
                             DataType dtypeA,
                             DataType dtypeB,
                             uint32_t M,
                             uint32_t K,
                             uint32_t N) {
  // Requires: fp16 inputs, dimensions aligned to 16, non-trivial size
  if (dtypeA != DataType::Float16 || dtypeB != DataType::Float16)
    return false;
  if (M % 16 != 0 || N % 16 != 0 || K % 16 != 0)
    return false;
  if (M < 16 || N < 16 || K < 16)
    return false;
  // The coopmat kernels assume 32-lane subgroups (the tiled kernel derives
  // 4 subgroups from its 128 threads; the others need one full 32-lane
  // subgroup). On wave64 devices (e.g. RADV) half the tiles would go
  // uncomputed, so fall back to the scalar variants there.
  if (caps.subgroupSize != 32)
    return false;
  // Check device capability (set by backend during init)
  if (!caps.cooperativeMatrix)
    return false;
  return true;
}

// Helper: check if cooperative matrix GEMV variant should be used for M=1.
// Relaxes the M>=16 requirement — only N and K need to be aligned to 16.
// A does not need to be Float16: resolveInputDtypes will insert a cast.
// At M=1 the cast is cheap (single row), and tensor cores are much faster.
static bool shouldUseCoopMatGemv(const DeviceCaps &caps,
                                 DataType dtypeB,
                                 uint32_t K,
                                 uint32_t N) {
  if (dtypeB != DataType::Float16)
    return false;
  if (N % 16 != 0 || K % 16 != 0)
    return false;
  if (N < 16 || K < 16)
    return false;
  // Needs one full 32-lane subgroup (see shouldUseCoopMat).
  if (caps.subgroupSize != 32)
    return false;
  if (!caps.cooperativeMatrix)
    return false;
  return true;
}

// ============================================================================
// Shape-based variant selection for standard (non-quantized) matmul.
//
// Built from autotune benchmark data (matmul_autotune.cpp) on RTX 3090.
// The heuristic groups shapes by M size and total work (M*K*N) to select
// the variant that performed best across representative shapes.
//
// Variant key:
//   1  = MatMul          (SharedMem 16x16, good for small matrices)
//   4  = MatMulT8R2x2    (T8 R2x2, good for small M + large K*N)
//   5  = MatMulT8R4x4    (T8 R4x4, default — good general-purpose)
//   7  = MatMulT16R8x8   (T16 R8x8, best for large matrices)
//   15 = MatMulVecBRegT16R4x4 (Vec4+BReg, good for medium-large)
//   19 = MatMulVecBRegAlignedT16R4x4 (Aligned vec4+BReg, K%16==0 N%64==0)
//   20 = MatMulGemv       (GEMV for M=1)
//   21 = MatMulGemv8      (split-K GEMV for M=1)
//   24 = MatMulGemv8M     (wave-per-row GEMV for M=2..16, N%8==0)
// ============================================================================

static int selectStandardVariant(uint32_t M, uint32_t K, uint32_t N) {
  constexpr int kGemv = 20;
  constexpr int kGemv8 = 21;           // MatMulGemv8 (8 cols/WG, K-unroll x4)
  constexpr int kGemv8M = 24;          // MatMulGemv8M (8 cols/WG, wave-per-row)
  constexpr int kSharedMem = 1;        // MatMul (SharedMem 16x16)
  constexpr int kT8R2x2 = 4;          // MatMulT8R2x2
  constexpr int kT8R4x4 = 5;          // MatMulT8R4x4 (current default)
  constexpr int kT16R8x8 = 7;         // MatMulT16R8x8
  constexpr int kVecBReg = 15;        // MatMulVecBRegT16R4x4
  constexpr int kVecBRegAligned = 19;  // MatMulVecBRegAlignedT16R4x4
  constexpr int kVecBRegAlignedK8 = 22; // MatMulVecBRegAlignedK8T16R4x4
  constexpr int kDblBuf = 17;         // MatMulDblBufT16R4x4
  constexpr int kDblBufVecAligned = 23; // MatMulDblBufVecAlignedT16R4x4

  // Aligned variant tile parameters (must match shaders.json)
  constexpr uint32_t kAlignedTileSize = 16;
  constexpr uint32_t kAlignedTN = 4;

  // GEMV: use Gemv8 when N is large enough to benefit from 8-col processing
  if (M == 1)
    return (N >= 8) ? kGemv8 : kGemv;

  uint64_t work = static_cast<uint64_t>(M) * K * N;

  // Check if dimensions qualify for the aligned variant:
  // K must be a multiple of TILE_SIZE (16) and N must be a multiple of
  // TILE_SIZE*TN (64). This eliminates partial-K tiles and N-edge
  // workgroups, allowing vec4 loads everywhere with minimal bounds checks.
  bool canUseAligned =
      (K % kAlignedTileSize == 0) && (N % (kAlignedTileSize * kAlignedTN) == 0);

  // Small M (2-16): shared-memory 16x16 wins for moderate K*N;
  // T8R2x2 wins when K*N is very large (bandwidth-bound).
  if (M <= 16) {
    // Experimental wave-per-row GEMV (opt-in): measured 2026-07-22 it does
    // NOT beat the shared-mem tile kernel at prefill shapes on the 3090
    // (Vulkan ~equal, CUDA 2-4x slower — the 8 waves' redundant B reads
    // thrash L1). Kept selectable for kernel experiments via env knob.
    static const bool kUseSmallMGemv = std::getenv("CUT_SMALLM_GEMV") != nullptr;
    if (kUseSmallMGemv && N % 8 == 0)
      return kGemv8M;
    if (work > 32 * 1024 * 1024)
      return kT8R2x2;
    return kSharedMem;
  }

  // Medium M (17-64): default T8R4x4 is already good.
  // SharedMem wins for small total work.
  // For deep K with aligned dims, use DblBufVecAligned (latency hiding + vec4).
  // Otherwise K8 aligned variant for aligned dims, DblBuf for unaligned.
  if (M <= 64) {
    if (work < 4 * 1024 * 1024)
      return kSharedMem;
    if (K > 1024 && work > 16 * 1024 * 1024)
      return canUseAligned ? kDblBufVecAligned : kDblBuf;
    return kT8R4x4;
  }

  // Large M (65-255): T16R8x8 starts winning for large K*N.
  // DblBufVecAligned for deep K with aligned dims (latency hiding dominant).
  // K8 aligned variant for medium work with aligned dims.
  if (M <= 255) {
    if (work > 128 * 1024 * 1024)
      return kT16R8x8;
    if (K > 1024 && work > 32 * 1024 * 1024)
      return canUseAligned ? kDblBufVecAligned : kDblBuf;
    if (work > 16 * 1024 * 1024)
      return canUseAligned ? kVecBRegAlignedK8 : kVecBReg;
    return kT8R4x4;
  }

  // Very large M (256+): T16R8x8 dominates.
  // DblBufVecAligned for deep K-loops with moderate total work and aligned dims.
  if (work > 8 * 1024 * 1024) {
    if (K > 1024 && work <= 64 * 1024 * 1024)
      return canUseAligned ? kDblBufVecAligned : kDblBuf;
    return kT16R8x8;
  }

  return kT8R4x4;
}

// Helper: map QuantFormat to the base OperatorEnum
static OperatorEnum matmulEnum(QuantFormat fmt) {
  switch (fmt) {
  case QuantFormat::Q8:
    return MatMulQ8;
  case QuantFormat::Q4:
    return MatMulQ4;
  default:
    return MatMul;
  }
}

// Helper: get variant info fields from the format-specific table
struct VariantInfo {
  uint32_t wgX, wgY, effTileM, effTileN;
};

static VariantInfo getVariant(QuantFormat fmt, uint32_t spec) {
  switch (fmt) {
  case QuantFormat::Q8: {
    const auto &v = kMatMulQ8Variants[spec];
    return {v.wgX, v.wgY, v.effTileM, v.effTileN};
  }
  case QuantFormat::Q4: {
    const auto &v = kMatMulQ4Variants[spec];
    return {v.wgX, v.wgY, v.effTileM, v.effTileN};
  }
  default: {
    const auto &v = kMatMulVariants[spec];
    return {v.wgX, v.wgY, v.effTileM, v.effTileN};
  }
  }
}

// ============================================================================
// Standard matmul constructor (2 inputs)
// ============================================================================

MatMulOpNode::MatMulOpNode(TensorStore &store,
                           const Tensor &a,
                           const Tensor &b,
                           std::optional<uint32_t> spec)
    : OpNode(MatMul, store, spec), format_(QuantFormat::None) {
  const auto &bufA = store.getTensor(a);
  const auto &bufB = store.getTensor(b);
  const auto shapeA = bufA.getShape();
  const auto shapeB = bufB.getShape();
  dtypeA_ = bufA.getDtype();
  dtypeB_ = bufB.getDtype();
  if (shapeB.size() != 2) {
    throw std::runtime_error("matmul requires a 2D B matrix");
  }
  // A may be 1D [K] (treated as [1,K]), matching the quantized constructor.
  // The graph's reshape passes strip the no-op [K]->[1,K] reshape, so fused
  // nodes (e.g. matmul+residual-add) must be constructible from a 1D A.
  if (shapeA.size() == 1) {
    M_ = 1;
    K_ = shapeA[0];
  } else if (shapeA.size() == 2) {
    M_ = shapeA[0];
    K_ = shapeA[1];
  } else {
    throw std::runtime_error("matmul: A must be 1D or 2D");
  }
  if (K_ != shapeB[0]) {
    throw std::runtime_error(
        "Matrix dimension mismatch: A is " + std::to_string(M_) + "x" +
        std::to_string(K_) + ", B is " + std::to_string(shapeB[0]) + "x" +
        std::to_string(shapeB[1]));
  }
  N_ = shapeB[1];
  autoSpec_ = !spec.has_value();
  fusionFallbackSpec_ = selectStandardVariant(M_, K_, N_);
  if (spec.has_value()) {
    spec_ = *spec;
  } else if (shouldUseCoopMat(store.caps(), dtypeA_, dtypeB_, M_, K_, N_)) {
    // Cooperative matrix variant — use tiled (2×2) for larger matrices
    spec_ = (M_ >= 32 && N_ >= 32) ? kCoopMatTiledVariant : kCoopMatVariant;
  } else if (dtypeA_ == DataType::Float32 && dtypeB_ == DataType::Float32) {
    // Data-driven per-backend selection (exact-shape tuning data); falls back
    // to the built-in shape heuristic (fusionFallbackSpec_) when no rule matches.
    int sel = VariantSelector::instance().select(
        "MatMul", {M_, K_, N_}, fusionFallbackSpec_,
        backendName(store.caps().backend));
    // A selected variant may still have no compiled shader for these operand
    // dtypes. Fall back to the built-in heuristic variant, which is always
    // available, if so. This mirrors the non-coopmat lookup in shader().
    if (!getCompiledMatMul(sel, dtypeA_, dtypeB_, dtypeA_).has_value())
      sel = fusionFallbackSpec_;
    spec_ = sel;
  } else {
    // Autotune measures MatMul with Float32 A and B and derives rules keyed on
    // {M, K, N} with no dtype condition, so applying them to another dtype pair
    // picks a variant that was never measured for it. The unquantized llama
    // path is Float32 activation x Float16 weight, where that cost up to 35% of
    // decode. Use the built-in heuristic, which accounts for the dtypes.
    spec_ = fusionFallbackSpec_;
  }
  inputs_ = {a, b};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

// ============================================================================
// Quantized matmul constructor (3 inputs, auto-detect Q4/Q8)
// ============================================================================

MatMulOpNode::MatMulOpNode(TensorStore &store,
                           const Tensor &a,
                           const Tensor &packedB,
                           const Tensor &scalesB,
                           std::optional<uint32_t> spec)
    : OpNode(MatMul, store, spec), format_(QuantFormat::None) {
  const auto &bufA = store.getTensor(a);
  const auto &bufPackedB = store.getTensor(packedB);
  const auto &bufScalesB = store.getTensor(scalesB);

  const auto shapeA = bufA.getShape();
  const auto shapePB = bufPackedB.getShape();
  const auto shapeSB = bufScalesB.getShape();

  dtypeA_ = bufA.getDtype();
  dtypeB_ = bufScalesB.getDtype(); // scales dtype (Float16 typically)

  if (shapePB.size() != 2 || shapeSB.size() != 2) {
    throw std::runtime_error("matmul requires 2D B and scales matrices");
  }
  // A can be 1D [K] (treated as [1, K]) — optimizer may remove the reshape
  if (shapeA.size() == 1) {
    M_ = 1;
    K_ = shapeA[0];
  } else if (shapeA.size() == 2) {
    M_ = shapeA[0];
    K_ = shapeA[1];
  } else {
    throw std::runtime_error("matmul: A must be 1D or 2D");
  }

  // Validate B rows match K
  if (shapePB[0] != K_) {
    throw std::runtime_error("matmul: B rows (" + std::to_string(shapePB[0]) +
                             ") must match A cols (" + std::to_string(K_) +
                             ")");
  }

  // Auto-detect Q4 vs Q8 from shapes:
  //   Q8:        B [K, N],   scales [K/32, N]   -> B.cols == scales.cols
  //   Q4:        B [K, N/2], scales [K/32, N]   -> B.cols * 2 == scales.cols
  //   Q4 affine: B [K, N/2], scales [2*K/32, N] -> as Q4, twice the rows; the
  //              lower half holds the per-block mins.
  uint32_t bCols = shapePB[1];
  uint32_t sCols = shapeSB[1];
  if (bCols == sCols) {
    format_ = QuantFormat::Q8;
    N_ = bCols;
  } else if (bCols * 2 == sCols) {
    format_ = QuantFormat::Q4;
    N_ = sCols;
    const uint32_t blocksK = (K_ + 31u) / 32u;
    q4Affine_ = (shapeSB[0] == 2u * blocksK);
  } else {
    throw std::runtime_error(
        "matmul: cannot determine quant format from B cols (" +
        std::to_string(bCols) + ") and scales cols (" + std::to_string(sCols) +
        ")");
  }

  // Set the correct OperatorEnum for this format
  op_ = matmulEnum(format_);

  autoSpec_ = !spec.has_value();
  if (format_ == QuantFormat::Q8) {
    if (M_ == 1) {
      // GEMV: 8-col K-parallel variant with K-unroll x4 (index 4)
      // Falls back to 4-col variant for very small N
      // Note: GemvDot (index 5) is NOT auto-selected — GEMV is memory-bound,
      // dotPacked4x8EXT overhead outweighs savings for M=1.
      fusionFallbackSpec_ = (N_ >= 8) ? 4 : 3; // GemvKPar8 or GemvKPar
    } else {
      // NOT TiledDot: it produces wrong results at this geometry (see
      // MatMulQ8_TiledDotVariant_Unfused_M33 test), so fused matmuls always
      // use the HLSL tiled variant whose writeOutput() fusion is validated.
      fusionFallbackSpec_ = kMatMulQ8DefaultVariant; // VecT16R4x4
    }
  } else {
    // Q4 (no cooperative-matrix variant exists)
    if (M_ == 1) {
      // GEMV: 8-col K-parallel variant with K-unroll x4
      // Falls back to 4-col variant for very small N
      // Absolute indices — these must not drift when a variant is appended
      // to the table.
      fusionFallbackSpec_ = (N_ >= 8) ? 3  // MatMulQ4GemvKPar8
                                      : 2; // MatMulQ4GemvKPar
    } else {
      fusionFallbackSpec_ = kMatMulQ4DefaultVariant;
    }
  }
  if (spec.has_value()) {
    spec_ = *spec;
  } else if (format_ == QuantFormat::Q8 && M_ > 1 &&
             store.caps().cooperativeMatrix && dtypeB_ == DataType::Float16 &&
             K_ % 16 == 0 && N_ % 32 == 0) {
    // CoopMat: dequant B→fp16 in shared memory, tensor core compute (best)
    spec_ = kMatMulQ8VariantCount - 1; // CoopMatTiled (last Q8 variant)
  } else if (format_ == QuantFormat::Q4 && M_ > 1 &&
             store.caps().cooperativeMatrix && dtypeB_ == DataType::Float16 &&
             K_ % 16 == 0 && N_ % 32 == 0) {
    // Same deal for Q4: unpack nibbles -> fp16 in shared memory, tensor cores.
    // Without this a Q4 weight falls to the scalar tiled variant and prefill
    // costs ~2x what the Q8 path does.
    spec_ = 4; // MatMulQ4CoopMatTiled
  } else {
    // NOTE: TiledDot (spec 6) is deliberately NOT auto-selected — it produces
    // wrong results even unfused (see MatMulQ8_TiledDotVariant_Unfused_M33,
    // currently DISABLED). Reachable via explicit spec only, until fixed.
    spec_ = fusionFallbackSpec_;
  }
  inputs_ = {a, packedB, scalesB};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

// ============================================================================
// Fusion support
// ============================================================================

void MatMulOpNode::setFusion(MatMulFusion fusion,
                             OperatorEnum fusionOp,
                             const Tensor &d) {
  fusion_ = fusion;
  fusionOp_ = fusionOp;
  if (fusion == MatMulFusion::Binary) {
    inputs_.push_back(d);
  }
  if (fusion == MatMulFusion::None) {
    return;
  }
  // The float cooperative-matrix shaders still have no fusion epilogue, so a
  // fused matmul would silently drop the fused op there and must fall back to
  // a scalar variant. The Q8 and Q4 cooperative-matrix tiled variants both
  // have one, so they are exempt.
  bool coopMatSelected = false;
  if (format_ == QuantFormat::None) {
    coopMatSelected = (*spec_ == kCoopMatVariant ||
                       *spec_ == kCoopMatTiledVariant ||
                       *spec_ == kCoopMatGemvVariant);
  }
  if (!coopMatSelected) {
    return;
  }
  if (!autoSpec_) {
    throw std::runtime_error(
        "matmul fusion is not supported by cooperative-matrix variants");
  }
  spec_ = fusionFallbackSpec_;
}

// ============================================================================
// Shared methods
// ============================================================================

DataType MatMulOpNode::outputDtype() const {
  return DataType::Float32;
}

size_t MatMulOpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(dtypeA_) & 0xF) << 16;
  key |= (static_cast<size_t>(dtypeB_) & 0xF) << 20;
  // Affine Q4 changes the compiled shader but is not part of spec_, so it
  // needs its own bit or the two variants collide in the shader cache.
  key |= (static_cast<size_t>(q4Affine_ ? 1 : 0)) << 35;
  // Encode fusion type (bits 36-39) and binary op (bits 40-47)
  key |= (static_cast<size_t>(fusion_) & 0xF) << 36;
  if (fusion_ != MatMulFusion::None) {
    key |= (static_cast<size_t>(fusionOp_) & 0xFF) << 40;
  }
  key |= static_cast<size_t>(spec_.value_or(0)) << 48;
  return key;
}

std::optional<std::vector<uint32_t>> MatMulOpNode::shader() const {
  // Get base matmul SPIR-V
  std::optional<std::vector<uint32_t>> compiled;
  switch (format_) {
  case QuantFormat::Q8:
    compiled = getCompiledMatMulQ8(*spec_, dtypeA_, dtypeB_, DataType::Float32);
    break;
  case QuantFormat::Q4:
    compiled = getCompiledMatMulQ4(*spec_, dtypeA_, dtypeB_, DataType::Float32);
    break;
  default:
    // CoopMat variants produce Float32 output from Float16 inputs
    if (*spec_ == kCoopMatVariant || *spec_ == kCoopMatTiledVariant ||
        *spec_ == kCoopMatGemvVariant) {
      compiled = getCompiledMatMul(*spec_, dtypeA_, dtypeB_, DataType::Float32);
    } else {
      compiled = getCompiledMatMul(*spec_, dtypeA_, dtypeB_, dtypeA_);
    }
    break;
  }

  const bool needsPatch = fusion_ != MatMulFusion::None || q4Affine_;
  if (!compiled.has_value() || !needsPatch) {
    return compiled;
  }

  auto spirv = std::move(compiled.value());

  // Patch specialization constants (single SPIR-V scan): 1/2 select the fusion
  // epilogue, 3 turns on the affine (zero-point) Q4 dequant. They are
  // independent, so the set is assembled rather than written literally.
  std::vector<std::pair<uint32_t, uint32_t>> specs;
  if (fusion_ == MatMulFusion::Unary) {
    specs.push_back({1, 1});
    specs.push_back({2, static_cast<uint32_t>(fusionOp_)});
  } else if (fusion_ == MatMulFusion::Binary) {
    specs.push_back({1, 2});
    specs.push_back({2, static_cast<uint32_t>(fusionOp_)});
  }
  if (q4Affine_) {
    specs.push_back({3, 1});
  }
  patchSpecConstants(spirv, specs);

  return spirv;
}

std::vector<uint32_t> MatMulOpNode::outputShape() const {
  return {M_, N_};
}

ThreadSize MatMulOpNode::dispatchSize() const {
  auto vi = getVariant(format_, *spec_);
  uint32_t gridX = ((N_ + vi.effTileN - 1) / vi.effTileN) * vi.wgX;
  uint32_t gridY = ((M_ + vi.effTileM - 1) / vi.effTileM) * vi.wgY;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> MatMulOpNode::pushConstants() const {
  uint32_t strideA = (K_ + 3) & ~3u;
  if (format_ == QuantFormat::None) {
    uint32_t strideB = (N_ + 3) & ~3u;
    uint32_t strideC = strideB;
    struct PushConstants {
      uint32_t M, K, N, strideA, strideB, strideC;
    } pc{M_, K_, N_, strideA, strideB, strideC};
    return toBytes(pc);
  }
  if (format_ == QuantFormat::Q4) {
    uint32_t strideBNpacked = (N_ / 2 + 3) & ~3u;
    uint32_t strideC = (N_ + 3) & ~3u;
    uint32_t scaleStride = (N_ + 3) & ~3u;
    struct PushConstants {
      uint32_t M, K, N, strideA, strideBNpacked, strideC, scaleStride;
    } pc{M_, K_, N_, strideA, strideBNpacked, strideC, scaleStride};
    return toBytes(pc);
  }
  // Q8
  uint32_t strideBN = (N_ + 3) & ~3u;
  uint32_t strideC = (N_ + 3) & ~3u;
  uint32_t scaleStride = (N_ + 3) & ~3u;
  struct PushConstants {
    uint32_t M, K, N, strideA, strideBN, strideC, scaleStride;
  } pc{M_, K_, N_, strideA, strideBN, strideC, scaleStride};
  return toBytes(pc);
}

std::vector<ComputeBinding> MatMulOpNode::bindings() const {
  std::vector<ComputeBinding> result;
  uint32_t idx = 0;

  // Bind all inputs (A, B, [scalesB], [D for Binary fusion])
  for (const auto &h : inputs_) {
    result.emplace_back(idx++, h);
  }

  // For non-Binary fusion, insert a dummy D binding (use output_ as dummy).
  // Binary fusion already has D as the last entry in inputs_.
  if (fusion_ != MatMulFusion::Binary) {
    result.emplace_back(idx++, output_);
  }

  // Bind output C
  if (output_) {
    result.emplace_back(idx++, output_);
  }

  // Push constants
  auto pc = pushConstants();
  if (!pc.empty()) {
    result.emplace_back(
        idx, DataReference(pc.data(), static_cast<uint32_t>(pc.size())));
  }
  return result;
}

std::string MatMulOpNode::displayName() const {
  std::string name;
  if (*spec_ == kCoopMatVariant) {
    name = "MatMulCoopMat";
  } else if (*spec_ == kCoopMatTiledVariant) {
    name = "MatMulCoopMatTiled";
  } else if (*spec_ == kCoopMatGemvVariant) {
    name = "MatMulCoopMatGemv";
  } else {
    switch (format_) {
    case QuantFormat::Q8:
      name = "MatMulQ8";
      break;
    case QuantFormat::Q4:
      name = "MatMulQ4";
      break;
    default:
      name = "MatMul";
      break;
    }
  }
  switch (fusion_) {
  case MatMulFusion::Unary:
    name += "Unary";
    break;
  case MatMulFusion::Binary:
    name += "Binary";
    break;
  default:
    break;
  }
  return name;
}

std::vector<DataType> MatMulOpNode::resolveInputDtypes(
    const std::vector<DataType> &inputDtypes) const {
  if (format_ != QuantFormat::None) {
    // Quantized: widen A to Float32, keep packedB and scales as-is
    DataType dtA = widenPrecision(inputDtypes[0]);
    std::vector<DataType> result = {dtA, inputDtypes[1], inputDtypes[2]};
    // Binary fusion has D as 4th input
    if (fusion_ == MatMulFusion::Binary && inputDtypes.size() > 3) {
      result.push_back(widenPrecision(inputDtypes[3]));
    }
    return result;
  }
  // Standard: try various dtype combinations
  DataType dtA = inputDtypes[0];
  DataType dtB = inputDtypes[1];

  // CoopMat variants require Float16 inputs → Float32 output
  if (*spec_ == kCoopMatVariant || *spec_ == kCoopMatTiledVariant ||
      *spec_ == kCoopMatGemvVariant) {
    std::vector<DataType> result = {DataType::Float16, DataType::Float16};
    if (fusion_ == MatMulFusion::Binary && inputDtypes.size() > 2) {
      result.push_back(widenPrecision(inputDtypes[2]));
    }
    return result;
  }

  std::vector<DataType> result;
  if (getCompiledMatMul(*spec_, dtA, dtB, dtA).has_value())
    result = {dtA, dtB};
  else {
    DataType wA = widenPrecision(dtA);
    DataType wB = widenPrecision(dtB);
    if (getCompiledMatMul(*spec_, dtA, wB, wB).has_value())
      result = {dtA, wB};
    else if (getCompiledMatMul(*spec_, wA, dtB, wA).has_value())
      result = {wA, dtB};
    else if (getCompiledMatMul(*spec_, wA, wB, wA).has_value())
      result = {wA, wB};
    else
      result = {DataType::Float32, DataType::Float32};
  }

  // Binary fusion has D as 3rd input for standard format
  if (fusion_ == MatMulFusion::Binary && inputDtypes.size() > 2) {
    result.push_back(widenPrecision(inputDtypes[2]));
  }
  return result;
}

} // namespace cut
