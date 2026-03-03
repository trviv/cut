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
        ThreadSize{((n + 255) / 256) * 256, 1, 1}, toBytes(initPC), true));
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
          std::vector<Tensor>{sortKeys, sortVals},
          ThreadSize{dispatchThreads, 1, 1}, toBytes(pc), true));
    }
  }

  if (needsPadding) {
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalBitonicCopyBack, DataType::Float32,
        std::vector<Tensor>{sortKeys, sortVals, keysHandle, valsHandle},
        ThreadSize{((numElements + 255) / 256) * 256, 1, 1},
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

    // Step 1: Histogram
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalRadixHistogram, DataType::Float32,
        std::vector<Tensor>{curKeys, histogram},
        ThreadSize{256 * groupCount, 1, 1}, toBytes(pc), true));

    // Step 2: Exclusive prefix scan on histogram (single thread)
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalScanUint, DataType::Float32, std::vector<Tensor>{histogram},
        ThreadSize{1, 1, 1}, toBytes(histSize), true));

    // Step 3: Scatter (single thread for stability)
    subOps_.push_back(std::make_unique<InternalOpNode>(
        InternalRadixScatter, DataType::Float32,
        std::vector<Tensor>{curKeys, curVals, dstKeys, dstVals, histogram},
        ThreadSize{1, 1, 1}, toBytes(pc), true));
  }
}

} // namespace cut
