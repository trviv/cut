#include "ScanOp.h"
#include "Shaders.h"        // kScanVariants / kScanVariantCount
#include "TensorStore.h"
#include "VariantSelector.h"

#include <string>

namespace cut {

PrefixScanOpNode::PrefixScanOpNode(OperatorEnum op,
                                   TensorStore &store,
                                   const Tensor &a,
                                   std::optional<uint32_t> spec)
    : OpNode(op, store, spec) {
  const auto &buf = store.getTensor(a);
  dtype_ = buf.getDtype();
  numElements_ = actualElementCount(buf.getShape());
  inputs_ = {a};
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType PrefixScanOpNode::outputDtype() const {
  return dtype_;
}

std::vector<uint32_t> PrefixScanOpNode::outputShape() const {
  return store_->getTensor(inputs_[0]).getShape();
}

bool PrefixScanOpNode::isMultiPass() const {
  return true;
}
size_t PrefixScanOpNode::executionSize() const {
  return numElements_;
}

ThreadSize PrefixScanOpNode::dispatchSize() const {
  return {0, 0, 0};
}
std::vector<uint8_t> PrefixScanOpNode::pushConstants() const {
  return {};
}

void PrefixScanOpNode::buildSubOperations() {
  // Single-pass decoupled look-back scan. One tile == BLOCK*IPT elements; the
  // items-per-thread (IPT) is an autotunable knob exposed as shader variants
  // (kScanVariants), each with eff_tile = (BLOCK, IPT). Bigger IPT = bigger tile
  // = faster, bounded by the device's shared memory. Backend-agnostic: the
  // internal op resolves to the native CUDA kernel on CUDA and HLSL on Vulkan.
  uint32_t numElements = static_cast<uint32_t>(numElements_);
  uint32_t isExclusive = (op_ == PrefixScanExclusiveSum) ? 1u : 0u;

  // --- Select the IPT variant ---------------------------------------------
  // 1) explicit spec_ (autotune sweep) -> use it.
  // 2) else VariantSelector rules (tuning_data.json, per shape + backend).
  // 3) else the hardware default: the largest IPT variant whose tile fits this
  //    device's shared memory. A selected variant that would overrun shared
  //    (e.g. tuning data captured on a larger GPU) falls back to that default.
  const uint32_t maxShared = store_->maxSharedMemoryPerBlock();
  const size_t scalarSize = dataTypeSize(dtype_);
  auto fits = [&](int vi) -> bool {
    // sData[BLOCK*IPT] + a small fixed overhead (per-warp totals + scalars);
    // the 256 B margin covers both backends' book-keeping shared memory.
    size_t need = static_cast<size_t>(kScanVariants[vi].effTileM) *
                      kScanVariants[vi].effTileN * scalarSize +
                  256u;
    return need <= maxShared;
  };

  int hwDefault = 0;
  for (int i = 0; i < kScanVariantCount; ++i)
    if (fits(i))
      hwDefault = i; // variants ascend in IPT; last fitting == largest fitting

  int variant;
  if (spec_.has_value()) {
    variant = static_cast<int>(*spec_);
  } else {
    const std::string backend =
        (store_->caps().backend == ComputeBackend::CUDA) ? "cuda" : "vulkan";
    variant = VariantSelector::instance().select("Scan", {numElements},
                                                 hwDefault, backend);
  }
  if (variant < 0 || variant >= kScanVariantCount || !fits(variant))
    variant = hwDefault;

  const uint32_t BLOCK = kScanVariants[variant].effTileM;
  const uint32_t IPT = kScanVariants[variant].effTileN;
  const uint32_t TILE = BLOCK * IPT;

  uint32_t numTiles = (numElements + TILE - 1) / TILE;
  if (numTiles == 0)
    numTiles = 1;

  Tensor inputHandle = inputs_[0];
  Tensor outputHandle = output_;

  // Per-tile look-back descriptors: [status | aggregate | inclusive] for each
  // tile plus a trailing dynamic tile-counter slot (see ScanDecoupled.cu).
  uint32_t stateSize = 3u * numTiles + 1u;
  Tensor state = store_->acquireTempBuffer(stateSize, DataType::UInt32);

  // Zero the descriptors (status = NOT_READY) and the tile counter. Temp buffers
  // come from a reuse pool and are not zeroed on acquisition.
  struct FillPC {
    uint32_t numElements;
    uint32_t fillValue;
  } fillPC{stateSize, 0u};
  subOps_.push_back(std::make_unique<InternalOpNode>(
      InternalFillUint, DataType::Float32, std::vector<Tensor>{state}, state,
      ThreadSize{stateSize, 1, 1}, toBytes(fillPC), true));

  // The scan itself: one workgroup per tile.
  struct ScanPC {
    uint32_t numElements;
    uint32_t isExclusive;
    uint32_t numTiles;
  } scanPC{numElements, isExclusive, numTiles};
  // dtype_ selects the Float32 / Int32 / UInt32 dtype; `variant` selects the IPT
  // (it MUST match the BLOCK/IPT used above for TILE/numTiles). The state buffer
  // and its FillUint zeroing stay UInt32 — it holds status flags and value
  // bit-patterns, not scalars.
  subOps_.push_back(std::make_unique<InternalOpNode>(
      InternalScanDecoupled, dtype_,
      std::vector<Tensor>{inputHandle, outputHandle, state}, outputHandle,
      ThreadSize{numTiles * BLOCK, 1, 1}, toBytes(scanPC), /*barrierAfter=*/false,
      static_cast<uint32_t>(variant)));
}

} // namespace cut
