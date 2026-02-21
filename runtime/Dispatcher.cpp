#include "Dispatcher.h"

#include "OpNode.h"
#include "Shaders.h"
#include <ComputeInterface.h>

#include <cstring>
#include <stdexcept>

namespace cut {

namespace {

/// Returns the next power of 2 >= n.
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

Dispatcher::Dispatcher(ComputeInterface *iface) : iface_(iface) {}

bool Dispatcher::encode(std::unique_ptr<OpNode> node) {
  if (!iface_) {
    throw std::runtime_error("Dispatcher::encode: ComputeInterface is null");
  }

  // Sort with 0 or 1 elements is a no-op (nothing to reorder)
  OperatorEnum op = node->op();
  if ((op == SortBitonic || op == SortRadix) && node->executionSize() <= 1) {
    return false;
  }

  // Multi-pass ops delegate to specialized methods
  if (node->isMultiPass()) {
    auto bindings = node->handleBindings();
    size_t execSize = node->executionSize();

    if (op == PrefixScanExclusiveSum || op == PrefixScanInclusiveSum) {
      encodePrefixScan(op, bindings, execSize);
    } else if (op == SortBitonic) {
      encodeBitonicSort(bindings, execSize);
    } else if (op == SortRadix) {
      encodeRadixSort(bindings, execSize);
    } else {
      // Multi-workgroup reduce
      encodeMultiWorkgroupReduce(op, bindings, execSize);
    }
    return true;
  }

  // Dim-reduce ops need internal shader lookup with spec constant patching
  if (node->isDimReduce()) {
    Tensor dimShader = getOrCreateDimReduceShader(
        node->baseReduceOp(), node->shaderDtype(), node->spec());
    auto bindings = node->handleBindings();
    auto pushData = node->pushConstants();
    ComputeDispatch dispatch(dimShader, node->dispatchSize(), bindings);
    dispatch.bindData(DataReference(pushData.data(), pushData.size()),
                      static_cast<uint32_t>(bindings.size()));
    iface_->encode(std::move(dispatch));
    return true;
  }

  // Standard single-dispatch ops: resolve shader here
  Tensor shader =
      getOrCreateShader(node->op(), node->shaderDtype(), node->spec());
  auto bindings = node->handleBindings();
  auto pushData = node->pushConstants();
  ComputeDispatch dispatch(shader, node->dispatchSize(), bindings);
  dispatch.bindData(DataReference(pushData.data(), pushData.size()),
                    static_cast<uint32_t>(bindings.size()));
  iface_->encode(std::move(dispatch));
  return true;
}

void Dispatcher::encodePrefixScan(OperatorEnum op,
                                  const std::vector<ComputeBinding> &bindings,
                                  size_t executionSize) {
  uint32_t numElements = static_cast<uint32_t>(executionSize);
  uint32_t isExclusive = (op == PrefixScanExclusiveSum) ? 1u : 0u;
  uint32_t groupCount = (numElements + 255) / 256;

  // Extract input and output handles
  Tensor inputHandle, outputHandle;
  for (const auto &b : bindings) {
    if (!b.isHandle())
      continue;
    if (b.index() == 0)
      inputHandle = b.getHandle();
    else if (b.index() == 1)
      outputHandle = b.getHandle();
  }

  struct ScanPC {
    uint32_t numElements;
    uint32_t isExclusive;
  } scanPC{numElements, isExclusive};

  if (groupCount <= 1) {
    // Single workgroup: simple scan, no temp buffers needed
    Tensor partialSums = acquireTempBuffer(1, DataType::Float32);
    dispatchInternal(InternalScanPerWg,
                     {{0u, inputHandle}, {1u, outputHandle}, {2u, partialSums}},
                     {256, 1, 1}, scanPC);
    releaseTempBuffers();
    return;
  }

  // Multi-workgroup: three-pass approach
  Tensor partialSums = acquireTempBuffer(groupCount, DataType::Float32);

  // Pass 1: Per-workgroup scan
  dispatchInternal(InternalScanPerWg,
                   {{0u, inputHandle}, {1u, outputHandle}, {2u, partialSums}},
                   {256 * groupCount, 1, 1}, scanPC);
  encodeBarrier();

  // Pass 2: Exclusive scan on partial sums (single thread)
  dispatchInternal(InternalScanPartialSums, {{0u, partialSums}}, {1, 1, 1},
                   groupCount);
  encodeBarrier();

  // Pass 3: Add group prefix to each element
  dispatchInternal(InternalScanPropagate,
                   {{0u, partialSums}, {1u, outputHandle}},
                   {256 * groupCount, 1, 1}, numElements);

  releaseTempBuffers();
}

void Dispatcher::encodeBitonicSort(const std::vector<ComputeBinding> &bindings,
                                   size_t executionSize) {
  uint32_t numElements = static_cast<uint32_t>(executionSize);
  if (numElements <= 1)
    return; // Nothing to sort
  uint32_t n = nextPowerOf2(numElements);

  // Extract keys and values handles
  Tensor keysHandle, valsHandle;
  for (const auto &b : bindings) {
    if (!b.isHandle())
      continue;
    if (b.index() == 0)
      keysHandle = b.getHandle();
    else if (b.index() == 1)
      valsHandle = b.getHandle();
  }

  // For non-power-of-2 sizes, pad to power-of-2 with sentinel values.
  // The bitonic network requires all N elements to participate in
  // compare-and-swap for correctness.
  Tensor sortKeys = keysHandle;
  Tensor sortVals = valsHandle;
  bool needsPadding = (numElements != n);

  if (needsPadding) {
    sortKeys = acquireTempBuffer(n, DataType::Float32);
    sortVals = acquireTempBuffer(n, DataType::UInt32);

    // Copy real data and fill padding with sentinels (FLT_MAX / 0xFFFFFFFF)
    struct InitPC {
      uint32_t numElements;
      uint32_t paddedSize;
    } initPC{numElements, n};
    dispatchInternal(
        InternalBitonicPadInit,
        {{0u, keysHandle}, {1u, valsHandle}, {2u, sortKeys}, {3u, sortVals}},
        {((n + 255) / 256) * 256, 1, 1}, initPC);
    encodeBarrier();
  }

  // Run bitonic sort on (possibly padded) buffers
  // Pre-fetch shader outside the O(log^2 N) loop
  Tensor stepShader = getOrCreateInternalShader(InternalBitonicStep);
  uint32_t dispatchThreads = ((n + 255) / 256) * 256;

  for (uint32_t k = 2; k <= n; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      struct StepPC {
        uint32_t numElements;
        uint32_t outerStep;
        uint32_t innerStep;
      } pc{n, k, j};
      dispatchInternal(stepShader, {{0u, sortKeys}, {1u, sortVals}},
                       {dispatchThreads, 1, 1}, pc);
      encodeBarrier();
    }
  }

  if (needsPadding) {
    // Copy sorted data back from padded temp buffers to user buffers
    dispatchInternal(
        InternalBitonicCopyBack,
        {{0u, sortKeys}, {1u, sortVals}, {2u, keysHandle}, {3u, valsHandle}},
        {((numElements + 255) / 256) * 256, 1, 1}, numElements);
    releaseTempBuffers();
  }
}

void Dispatcher::encodeRadixSort(const std::vector<ComputeBinding> &bindings,
                                 size_t executionSize) {
  uint32_t numElements = static_cast<uint32_t>(executionSize);
  if (numElements <= 1)
    return; // Nothing to sort

  // Extract keys and values handles
  Tensor keysHandle, valsHandle;
  for (const auto &b : bindings) {
    if (!b.isHandle())
      continue;
    if (b.index() == 0)
      keysHandle = b.getHandle();
    else if (b.index() == 1)
      valsHandle = b.getHandle();
  }

  uint32_t groupCount = std::max((numElements + 255) / 256, 1u);
  uint32_t histSize = 16 * groupCount; // 16 digits * groupCount

  Tensor histogram = acquireTempBuffer(histSize, DataType::UInt32);
  Tensor keysAlt = acquireTempBuffer(numElements, DataType::UInt32);
  Tensor valsAlt = acquireTempBuffer(numElements, DataType::UInt32);

  // Pre-fetch shaders outside the loop
  Tensor histShader = getOrCreateInternalShader(InternalRadixHistogram);
  Tensor scatterShader = getOrCreateInternalShader(InternalRadixScatter);
  Tensor scanUintShader = getOrCreateInternalShader(InternalScanUint);

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
    dispatchInternal(histShader, {{0u, curKeys}, {1u, histogram}},
                     {256 * groupCount, 1, 1}, pc);
    encodeBarrier();

    // Step 2: Exclusive prefix scan on histogram (single thread)
    dispatchInternal(scanUintShader, {{0u, histogram}}, {1, 1, 1}, histSize);
    encodeBarrier();

    // Step 3: Scatter (single thread for stability)
    dispatchInternal(scatterShader,
                     {{0u, curKeys},
                      {1u, curVals},
                      {2u, dstKeys},
                      {3u, dstVals},
                      {4u, histogram}},
                     {1, 1, 1}, pc);
    encodeBarrier();
  }

  // After 8 passes (even number), final result is back in keysHandle/valsHandle
  releaseTempBuffers();
}

void Dispatcher::encodeMultiWorkgroupReduce(
    OperatorEnum op,
    const std::vector<ComputeBinding> &bindings,
    size_t executionSize) {
  // Infer dtype from bindings
  DataType dtype = ComputeBuffer::inferDataType(
      bindings, [this](const Tensor &h) -> const ComputeBuffer & {
        return iface_->getBuffer(h);
      });

  uint32_t numElements = static_cast<uint32_t>(executionSize);

  // Each WG of 256 threads processes ~1024 elements, cap at 256 workgroups
  uint32_t groupCount = (numElements + 1023) / 1024;
  groupCount = std::min(groupCount, 256u);
  groupCount = std::max(groupCount, 2u);

  // Extract input and output handles from bindings
  Tensor inputHandle, outputHandle;
  for (const auto &b : bindings) {
    if (!b.isHandle())
      continue;
    if (b.index() == 0)
      inputHandle = b.getHandle();
    else if (b.index() == 1)
      outputHandle = b.getHandle();
  }

  Tensor partialShader =
      getOrCreateInternalShader(InternalPartialReduce, dtype);
  Tensor finalShader = getOrCreateInternalShader(InternalFinalReduce, dtype);

  Tensor partialSums = acquireTempBuffer(groupCount, dtype);

  // Phase 1: Partial reduce — each workgroup reduces its batch
  struct PartialPC {
    uint32_t numElements;
    uint32_t groupCount;
    uint32_t reduceOp;
  } partialPC{numElements, groupCount, static_cast<uint32_t>(op)};
  dispatchInternal(partialShader, {{0u, inputHandle}, {1u, partialSums}},
                   {256 * groupCount, 1, 1}, partialPC);
  encodeBarrier();

  // Phase 2: Final reduce — single workgroup reduces partial sums
  struct FinalPC {
    uint32_t numElements;
    uint32_t originalNumElements;
    uint32_t reduceOp;
  } finalPC{groupCount, numElements, static_cast<uint32_t>(op)};
  dispatchInternal(finalShader, {{0u, partialSums}, {1u, outputHandle}},
                   {256, 1, 1}, finalPC);

  releaseTempBuffers();
}

Tensor Dispatcher::acquireTempBuffer(size_t numElements, DataType dtype) {
  // Calculate aligned size in bytes for pool lookup
  size_t sizeBytes = ComputeBuffer::calculateAlignedSize(
      {static_cast<uint32_t>(numElements)}, dtype);

  // Iterate pool to find a buffer of sufficient size
  for (auto it = tempBufferPool_.begin(); it != tempBufferPool_.end(); ++it) {
    const auto &buffer = iface_->getBuffer(*it);
    if (buffer.size() >= sizeBytes) {
      Tensor handle = *it;
      tempBufferPool_.erase(it);
      activeTempBuffers_.push_back(handle);
      return handle;
    }
  }

  // No pooled buffer available — create a new one
  Tensor handle =
      iface_->createBuffer({static_cast<uint32_t>(numElements)}, dtype);
  activeTempBuffers_.push_back(handle);
  return handle;
}

void Dispatcher::releaseTempBuffers() {
  for (const auto &handle : activeTempBuffers_) {
    tempBufferPool_.push_back(handle);
  }
  activeTempBuffers_.clear();
}

void Dispatcher::encodeBarrier() {
  iface_->encode(ComputeDispatch::createBarrier());
}

void Dispatcher::dispatchInternal(const Tensor &shader,
                                  const std::vector<ComputeBinding> &bindings,
                                  ThreadSize threadSize,
                                  const DataReference &pushData) {
  ComputeDispatch dispatch(shader, threadSize, bindings);
  dispatch.bindData(pushData, static_cast<uint32_t>(bindings.size()));
  iface_->encode(std::move(dispatch));
}

void Dispatcher::dispatchInternal(OperatorEnum op,
                                  const std::vector<ComputeBinding> &bindings,
                                  ThreadSize threadSize,
                                  const DataReference &pushData) {
  dispatchInternal(getOrCreateInternalShader(op), bindings, threadSize,
                   pushData);
}

Tensor Dispatcher::getOrCreateInternalShader(OperatorEnum op, DataType dtype) {
  size_t key = static_cast<size_t>(op) | (static_cast<size_t>(dtype) << 16) |
               (size_t(1) << 48);

  auto it = internalShaderCache_.find(key);
  if (it != internalShaderCache_.end()) {
    return it->second;
  }

  // Compile via the shader generation system
  auto spirv = getShader(op, dtype);
  Tensor handle = iface_->createShaderModule(spirv);
  internalShaderCache_[key] = handle;
  return handle;
}

Tensor Dispatcher::getOrCreateShader(OperatorEnum op,
                                     DataType dtype,
                                     std::optional<uint32_t> spec) {
  // Use (3 << 48) prefix to distinguish from internal/dim-reduce shaders
  size_t key = static_cast<size_t>(op) | (static_cast<size_t>(dtype) << 16) |
               (size_t(3) << 48) |
               (static_cast<size_t>(spec.value_or(0)) << 32);

  auto it = internalShaderCache_.find(key);
  if (it != internalShaderCache_.end()) {
    return it->second;
  }

  Tensor shader;
  if (spec.has_value()) {
    uint32_t s = spec.value();
    std::optional<std::vector<uint32_t>> spirv;
    switch (op) {
    case MatMul:
      spirv = getCompiledMatMul(s, dtype);
      break;
    case Transpose:
      spirv = getCompiledTranspose(s, dtype);
      break;
    case Conv1D:
      spirv = getCompiledConv1D(s, dtype);
      break;
    case Conv2D:
      spirv = getCompiledConv2D(s, dtype);
      break;
    case MaxPool2D:
      spirv = getCompiledMaxPool2D(s, dtype);
      break;
    case AvgPool2D:
      spirv = getCompiledAvgPool2D(s, dtype);
      break;
    default:
      throw std::runtime_error("No spec support for op " + std::to_string(op));
    }
    if (!spirv.has_value()) {
      throw std::runtime_error("Failed to get spec " + std::to_string(s) +
                               " for op " + std::to_string(op));
    }
    shader = iface_->createShaderModule(spirv.value());
  } else {
    std::vector<uint32_t> spirv = getShader(op, dtype);
    shader = iface_->createShaderModule(spirv);
  }

  internalShaderCache_[key] = shader;
  return shader;
}

Tensor Dispatcher::getOrCreateDimReduceShader(OperatorEnum reduceOp,
                                              DataType dtype,
                                              std::optional<uint32_t> spec) {
  // Use bit 49 to distinguish dim-reduce shaders from global-reduce shaders
  // Include spec in the cache key (bits 32-47)
  size_t key = static_cast<size_t>(reduceOp) |
               (static_cast<size_t>(dtype) << 16) | (size_t(2) << 48) |
               (static_cast<size_t>(spec.value_or(0)) << 32);

  auto it = internalShaderCache_.find(key);
  if (it != internalShaderCache_.end()) {
    return it->second;
  }

  auto spirv = getDimReduceShader(reduceOp, dtype, spec);
  Tensor handle = iface_->createShaderModule(spirv);
  internalShaderCache_[key] = handle;
  return handle;
}

} // namespace cut
