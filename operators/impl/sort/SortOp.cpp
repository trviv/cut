#include "SortOp.h"
#include "TensorStore.h"

namespace cut {

namespace {

uint32_t nextPowerOf2(uint32_t n) {
  if (n <= 1)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

// Builds the OneSweep radix-sort dispatch graph (one global-histogram pass over
// all 4 digit-places + per-pass exclusive scan + 4 decoupled-look-back scatter
// passes). Backend-agnostic: the internal ops resolve to native CUDA kernels on
// CUDA and to HLSL kernels on Vulkan. Used by both RadixSinglePassSortOpNode
// (CUDA branch) and RadixOneSweepSortOpNode (both backends).
void buildOneSweepGraph(TensorStore *store,
                        const Tensor &keysHandle,
                        const Tensor &valsHandle,
                        uint32_t numElements,
                        std::vector<std::unique_ptr<OpNode>> &subOps) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t NUM_PASSES = 4;
  constexpr uint32_t BLOCK = 256;
  // The OneSweep scatter kernel processes one element per thread, so a tile is
  // exactly BLOCK elements (idx = tile * BLOCK + tid). Keep in lockstep.
  constexpr uint32_t ELEMS_PER_TILE = BLOCK;

  uint32_t numTiles =
      std::max((numElements + ELEMS_PER_TILE - 1) / ELEMS_PER_TILE, 1u);

  Tensor globalHist =
      store->acquireTempBuffer(NUM_PASSES * RADIX, DataType::UInt32);
  Tensor keysAlt = store->acquireTempBuffer(numElements, DataType::UInt32);
  Tensor valsAlt = store->acquireTempBuffer(numElements, DataType::UInt32);
  // Decoupled look-back partition descriptors (one word per (tile, digit)) plus
  // a trailing slot used as the dynamic partition/tile counter. Each pass gets
  // its own contiguous region so all NUM_PASSES regions can be zeroed in a
  // single up-front dispatch (saves 3 fills + 3 barriers vs per-pass resets);
  // the scatter kernel derives its region base from passIndex.
  uint32_t stateStride = numTiles * RADIX + 1;
  Tensor lookbackState =
      store->acquireTempBuffer(NUM_PASSES * stateStride, DataType::UInt32);

  struct FillPC {
    uint32_t numElements;
    uint32_t fillValue;
  };

  // 1. Zero the global histogram (accumulated with atomics).
  {
    FillPC pc{NUM_PASSES * RADIX, 0u};
    subOps.push_back(std::make_unique<InternalOpNode>(
        InternalFillUint, DataType::Float32, std::vector<Tensor>{globalHist},
        globalHist, ThreadSize{NUM_PASSES * RADIX, 1, 1}, toBytes(pc), true));
  }

  // 2. Zero all NUM_PASSES look-back regions in one dispatch.
  {
    FillPC pc{NUM_PASSES * stateStride, 0u};
    subOps.push_back(std::make_unique<InternalOpNode>(
        InternalFillUint, DataType::Float32,
        std::vector<Tensor>{lookbackState}, lookbackState,
        ThreadSize{NUM_PASSES * stateStride, 1, 1}, toBytes(pc), true));
  }

  // 3. Global histogram over all NUM_PASSES digit-places (from original keys;
  //    digit-place counts are order-independent, so this is computed once).
  {
    struct HistPC {
      uint32_t numElements;
    } pc{numElements};
    subOps.push_back(std::make_unique<InternalOpNode>(
        InternalOneSweepGlobalHist, DataType::Float32,
        std::vector<Tensor>{keysHandle, globalHist}, globalHist,
        ThreadSize{numTiles * BLOCK, 1, 1}, toBytes(pc), true));
  }

  // 4. Per-pass exclusive scan of the global histogram (4 independent 256-wide
  //    scans; running sum resets at each pass boundary).
  subOps.push_back(std::make_unique<InternalOpNode>(
      InternalOneSweepGlobalScan, DataType::Float32,
      std::vector<Tensor>{globalHist}, globalHist, ThreadSize{1, 1, 1},
      std::vector<uint8_t>{}, true));

  // 5. Four decoupled-look-back scatter passes (ping-pong; 4 passes lands the
  //    result back in the caller's original keys/vals handles).
  for (uint32_t pass = 0; pass < NUM_PASSES; pass++) {
    bool evenPass = (pass % 2 == 0);
    Tensor curKeys = evenPass ? keysHandle : keysAlt;
    Tensor curVals = evenPass ? valsHandle : valsAlt;
    Tensor dstKeys = evenPass ? keysAlt : keysHandle;
    Tensor dstVals = evenPass ? valsAlt : valsHandle;

    struct ScatterPC {
      uint32_t numElements;
      uint32_t passIndex;
      uint32_t numTiles;
    } pc{numElements, pass, numTiles};
    subOps.push_back(std::make_unique<InternalOpNode>(
        InternalOneSweepScatter, DataType::Float32,
        std::vector<Tensor>{curKeys, curVals, dstKeys, dstVals, globalHist,
                            lookbackState},
        dstKeys, ThreadSize{numTiles * BLOCK, 1, 1}, toBytes(pc), true));
  }
}

} // namespace

// --- BitonicSortOpNode ---

BitonicSortOpNode::BitonicSortOpNode(TensorStore &store,
                                     const Tensor &keys,
                                     const Tensor &vals,
                                     std::optional<uint32_t> spec)
    : OpNode(SortBitonic, store, spec) {
  const auto &buf = store.getTensor(keys);
  executionSize_ = actualElementCount(buf.getShape());
  dtype_ = buf.getDtype();
  inputs_ = {keys, vals};
}

DataType BitonicSortOpNode::outputDtype() const {
  return dtype_;
}

std::vector<uint32_t> BitonicSortOpNode::outputShape() const {
  return {};
}

bool BitonicSortOpNode::isMultiPass() const {
  return true;
}
size_t BitonicSortOpNode::executionSize() const {
  return executionSize_;
}

ThreadSize BitonicSortOpNode::dispatchSize() const {
  return {0, 0, 0};
}
std::vector<uint8_t> BitonicSortOpNode::pushConstants() const {
  return {};
}

void BitonicSortOpNode::buildSubOperations() {
  uint32_t numElements = static_cast<uint32_t>(executionSize_);
  uint32_t n = nextPowerOf2(numElements);

  Tensor keysHandle = inputs_[0];
  Tensor valsHandle = inputs_[1];

  Tensor sortKeys = keysHandle;
  Tensor sortVals = valsHandle;
  bool needsPadding = (numElements != n);

  if (needsPadding) {
    sortKeys = store_->acquireTempBuffer(n, DataType::Float32);
    sortVals = store_->acquireTempBuffer(n, DataType::UInt32);

    struct InitPC {
      uint32_t numElements;
      uint32_t paddedSize;
    } initPC{numElements, n};
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalBitonicPadInit, DataType::Float32,
        std::vector<Tensor>{keysHandle, valsHandle, sortKeys, sortVals},
        sortKeys, ThreadSize{((n + 255) / 256) * 256, 1, 1}, toBytes(initPC),
        true));
  }

  uint32_t dispatchThreads = ((n + 255) / 256) * 256;

  for (uint32_t k = 2; k <= n; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      struct StepPC {
        uint32_t numElements;
        uint32_t outerStep;
        uint32_t innerStep;
      } pc{n, k, j};
      subOps_.push_back(std::make_unique<InternalOpNode>(
          InternalBitonicStep, DataType::Float32,
          std::vector<Tensor>{sortKeys, sortVals}, sortKeys,
          ThreadSize{dispatchThreads, 1, 1}, toBytes(pc), true));
    }
  }

  if (needsPadding) {
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalBitonicCopyBack, DataType::Float32,
        std::vector<Tensor>{sortKeys, sortVals, keysHandle, valsHandle},
        keysHandle, ThreadSize{((numElements + 255) / 256) * 256, 1, 1},
        toBytes(numElements)));
  }
}

// --- RadixSortOpNode ---

RadixSortOpNode::RadixSortOpNode(TensorStore &store,
                                 const Tensor &keys,
                                 const Tensor &vals,
                                 std::optional<uint32_t> spec)
    : OpNode(SortRadix, store, spec) {
  const auto &buf = store.getTensor(keys);
  executionSize_ = actualElementCount(buf.getShape());
  dtype_ = buf.getDtype();
  inputs_ = {keys, vals};
}

DataType RadixSortOpNode::outputDtype() const {
  return dtype_;
}

std::vector<uint32_t> RadixSortOpNode::outputShape() const {
  return {};
}

bool RadixSortOpNode::isMultiPass() const {
  return true;
}
size_t RadixSortOpNode::executionSize() const {
  return executionSize_;
}

ThreadSize RadixSortOpNode::dispatchSize() const {
  return {0, 0, 0};
}
std::vector<uint8_t> RadixSortOpNode::pushConstants() const {
  return {};
}

void RadixSortOpNode::buildSubOperations() {
  uint32_t numElements = static_cast<uint32_t>(executionSize_);

  Tensor keysHandle = inputs_[0];
  Tensor valsHandle = inputs_[1];

  uint32_t groupCount = std::max((numElements + 255) / 256, 1u);
  uint32_t histSize = 16 * groupCount;

  Tensor histogram = store_->acquireTempBuffer(histSize, DataType::UInt32);
  Tensor keysAlt = store_->acquireTempBuffer(numElements, DataType::UInt32);
  Tensor valsAlt = store_->acquireTempBuffer(numElements, DataType::UInt32);

  // 8 passes (4 bits each) for 32-bit keys
  for (uint32_t pass = 0; pass < 8; pass++) {
    uint32_t bitOffset = pass * 4;
    bool evenPass = (pass % 2 == 0);

    Tensor curKeys = evenPass ? keysHandle : keysAlt;
    Tensor curVals = evenPass ? valsHandle : valsAlt;
    Tensor dstKeys = evenPass ? keysAlt : keysHandle;
    Tensor dstVals = evenPass ? valsAlt : valsHandle;

    struct RadixPC {
      uint32_t numElements;
      uint32_t bitOffset;
      uint32_t groupCount;
    } pc{numElements, bitOffset, groupCount};

    // Step 1: Histogram — reads curKeys, writes histogram
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalRadixHistogram, DataType::Float32,
        std::vector<Tensor>{curKeys, histogram}, histogram,
        ThreadSize{256 * groupCount, 1, 1}, toBytes(pc), true));

    // Step 2: Exclusive prefix scan on histogram (in-place)
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalScanUint, DataType::Float32, std::vector<Tensor>{histogram},
        histogram, ThreadSize{1, 1, 1}, toBytes(histSize), true));

    // Step 3: Scatter — reads curKeys, curVals, histogram; writes dstKeys,
    // dstVals
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalRadixScatter, DataType::Float32,
        std::vector<Tensor>{curKeys, curVals, dstKeys, dstVals, histogram},
        dstKeys, ThreadSize{1, 1, 1}, toBytes(pc), true));
  }
}

// --- RadixSinglePassSortOpNode ---

RadixSinglePassSortOpNode::RadixSinglePassSortOpNode(
    TensorStore &store,
    const Tensor &keys,
    const Tensor &vals,
    std::optional<uint32_t> spec)
    : OpNode(SortRadixSinglePass, store, spec) {
  const auto &buf = store.getTensor(keys);
  executionSize_ = actualElementCount(buf.getShape());
  dtype_ = buf.getDtype();
  inputs_ = {keys, vals};
}

DataType RadixSinglePassSortOpNode::outputDtype() const {
  return dtype_;
}

std::vector<uint32_t> RadixSinglePassSortOpNode::outputShape() const {
  return {};
}

bool RadixSinglePassSortOpNode::isMultiPass() const {
  return true;
}
size_t RadixSinglePassSortOpNode::executionSize() const {
  return executionSize_;
}

ThreadSize RadixSinglePassSortOpNode::dispatchSize() const {
  return {0, 0, 0};
}
std::vector<uint8_t> RadixSinglePassSortOpNode::pushConstants() const {
  return {};
}

void RadixSinglePassSortOpNode::buildSubOperations() {
  // Same public op, backend-specialized dispatch graph: OneSweep decoupled
  // look-back on CUDA, fused per-digit tile radix on Vulkan.
  if (store_->caps().backend == ComputeBackend::CUDA) {
    buildOneSweepGraph(store_, inputs_[0], inputs_[1],
                       static_cast<uint32_t>(executionSize_), subOps_);
  } else {
    buildFusedVulkan();
  }
}

// Vulkan fused per-digit: parallel tile histogram (digit-major) -> exclusive
// spine scan -> per-tile local stable sort + parallel global scatter. 8-bit
// digits, 4 passes.
void RadixSinglePassSortOpNode::buildFusedVulkan() {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t NUM_PASSES = 4;
  constexpr uint32_t WG = 256;
  constexpr uint32_t ELEMS_PER_TILE = 256;

  uint32_t numElements = static_cast<uint32_t>(executionSize_);
  uint32_t numTiles =
      std::max((numElements + ELEMS_PER_TILE - 1) / ELEMS_PER_TILE, 1u);

  Tensor keysHandle = inputs_[0];
  Tensor valsHandle = inputs_[1];

  // Digit-major layout tileHist[digit * numTiles + tile]: a single exclusive
  // scan over the whole spine yields each (digit, tile) global base offset.
  Tensor tileHist =
      store_->acquireTempBuffer(RADIX * numTiles, DataType::UInt32);
  Tensor keysAlt = store_->acquireTempBuffer(numElements, DataType::UInt32);
  Tensor valsAlt = store_->acquireTempBuffer(numElements, DataType::UInt32);

  for (uint32_t pass = 0; pass < NUM_PASSES; pass++) {
    uint32_t bitOffset = pass * 8;
    bool evenPass = (pass % 2 == 0);
    Tensor curKeys = evenPass ? keysHandle : keysAlt;
    Tensor curVals = evenPass ? valsHandle : valsAlt;
    Tensor dstKeys = evenPass ? keysAlt : keysHandle;
    Tensor dstVals = evenPass ? valsAlt : valsHandle;

    struct DigitPC {
      uint32_t numElements;
      uint32_t bitOffset;
      uint32_t numTiles;
    } pc{numElements, bitOffset, numTiles};

    // 1. Tile histogram: each block owns one tile, writes all RADIX bins.
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalFusedTileHist, DataType::Float32,
        std::vector<Tensor>{curKeys, tileHist}, tileHist,
        ThreadSize{numTiles * WG, 1, 1}, toBytes(pc), true));

    // 2. Exclusive scan of the digit-major spine (reuse serial ScanUint).
    uint32_t spineSize = RADIX * numTiles;
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalScanUint, DataType::Float32, std::vector<Tensor>{tileHist},
        tileHist, ThreadSize{1, 1, 1}, toBytes(spineSize), true));

    // 3. Fused local stable sort + global scatter.
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalFusedScatter, DataType::Float32,
        std::vector<Tensor>{curKeys, curVals, dstKeys, dstVals, tileHist},
        dstKeys, ThreadSize{numTiles * WG, 1, 1}, toBytes(pc), true));
  }
}

// --- RadixOneSweepSortOpNode ---

RadixOneSweepSortOpNode::RadixOneSweepSortOpNode(TensorStore &store,
                                                 const Tensor &keys,
                                                 const Tensor &vals,
                                                 std::optional<uint32_t> spec)
    : OpNode(SortRadixOneSweep, store, spec) {
  const auto &buf = store.getTensor(keys);
  executionSize_ = actualElementCount(buf.getShape());
  dtype_ = buf.getDtype();
  inputs_ = {keys, vals};
}

DataType RadixOneSweepSortOpNode::outputDtype() const {
  return dtype_;
}

std::vector<uint32_t> RadixOneSweepSortOpNode::outputShape() const {
  return {};
}

bool RadixOneSweepSortOpNode::isMultiPass() const {
  return true;
}
size_t RadixOneSweepSortOpNode::executionSize() const {
  return executionSize_;
}

ThreadSize RadixOneSweepSortOpNode::dispatchSize() const {
  return {0, 0, 0};
}
std::vector<uint8_t> RadixOneSweepSortOpNode::pushConstants() const {
  return {};
}

void RadixOneSweepSortOpNode::buildSubOperations() {
  // OneSweep decoupled look-back on both backends (native CUDA kernels on CUDA,
  // HLSL kernels on Vulkan).
  buildOneSweepGraph(store_, inputs_[0], inputs_[1],
                     static_cast<uint32_t>(executionSize_), subOps_);
}

} // namespace cut
