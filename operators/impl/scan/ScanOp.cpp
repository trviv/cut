#include "ScanOp.h"
#include "TensorStore.h"

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
  // Single-pass decoupled look-back scan. One tile == BLOCK*IPT elements; these
  // MUST match the ScanDecoupled kernels' compile-time BLOCK/IPT. Backend-
  // agnostic: the internal op resolves to the native CUDA kernel on CUDA and to
  // the HLSL kernel on Vulkan.
  constexpr uint32_t BLOCK = 256;
  constexpr uint32_t IPT = 32;
  constexpr uint32_t TILE = BLOCK * IPT; // 8192

  uint32_t numElements = static_cast<uint32_t>(numElements_);
  uint32_t isExclusive = (op_ == PrefixScanExclusiveSum) ? 1u : 0u;
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
  // dtype_ selects the Float32 / Int32 / UInt32 kernel variant. (The state
  // buffer and its FillUint zeroing stay UInt32 regardless — it holds status
  // flags and value bit-patterns, not scalars.)
  subOps_.push_back(std::make_unique<InternalOpNode>(
      InternalScanDecoupled, dtype_,
      std::vector<Tensor>{inputHandle, outputHandle, state}, outputHandle,
      ThreadSize{numTiles * BLOCK, 1, 1}, toBytes(scanPC)));
}

} // namespace cut
