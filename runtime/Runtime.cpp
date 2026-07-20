#include "Runtime.h"

#include "Dispatcher.h"
#include "OpNode.h"
#include "Operations.h"
#include "VariantSelector.h"
#include "VulkanCompute.h"

#ifdef CUT_ENABLE_CUDA
#include "CudaCompute.h"
#endif

#include <ComputeCommon.h>
#include <chrono>
#include <stdexcept>

namespace cut {

// DeviceContext special members live here so the unique_ptr members can use
// types that are only forward-declared in the header.
Runtime::DeviceContext::DeviceContext() = default;
Runtime::DeviceContext::~DeviceContext() = default;
Runtime::DeviceContext::DeviceContext(DeviceContext &&) noexcept = default;
Runtime::DeviceContext &
Runtime::DeviceContext::operator=(DeviceContext &&) noexcept = default;

Runtime::Runtime() = default;

Runtime::~Runtime() {
  shutdown();
}

bool Runtime::isVulkanAvailable() {
  if (!vulkanChecked_) {
    try {
      vulkanInstance_ = std::make_shared<VulkanInstance>();
      vulkanAvailable_ = true;
    } catch (...) {
      vulkanAvailable_ = false;
    }
    vulkanChecked_ = true;
  }
  return vulkanAvailable_;
}

bool Runtime::isCudaAvailable() {
#ifdef CUT_ENABLE_CUDA
  if (!cudaChecked_) {
    try {
      cudaInstance_ = std::make_shared<CudaInstance>();
      cudaAvailable_ = true;
    } catch (...) {
      cudaAvailable_ = false;
    }
    cudaChecked_ = true;
  }
  return cudaAvailable_;
#else
  return false;
#endif
}

void Runtime::init(BackendType backend) {
  init(std::vector<DeviceDesc>{DeviceDesc{backend, -1}});
}

void Runtime::init(const std::vector<DeviceDesc> &devices) {
  if (devices.empty()) {
    throw std::runtime_error("Cannot initialize with an empty device list");
  }
  if (!devices_.empty()) {
    throw std::runtime_error("Runtime already initialized");
  }

  try {
    for (const auto &desc : devices) {
      DeviceContext ctx;
      ctx.backend = desc.backend;

      if (desc.backend == BackendType::Vulkan) {
        if (!isVulkanAvailable()) {
          throw std::runtime_error("Vulkan backend is not available");
        }
        VulkanContextConfig cfg;
        cfg.preferredDevice = desc.deviceIndex;
        ctx.interface = vulkanInstance_->createInterface(cfg);
      } else if (desc.backend == BackendType::CUDA) {
#ifdef CUT_ENABLE_CUDA
        if (!isCudaAvailable()) {
          throw std::runtime_error("CUDA backend is not available");
        }
        CudaContextConfig cfg;
        cfg.preferredDevice = desc.deviceIndex;
        ctx.interface = cudaInstance_->createInterface(cfg);
#else
        throw std::runtime_error("CUDA backend not compiled in "
                                 "(rebuild with -DENABLE_CUDA_BACKEND=ON)");
#endif
      } else {
        throw std::runtime_error("Invalid backend type");
      }

      ctx.store = std::make_unique<TensorStore>(ctx.interface.get());
      ctx.dispatcher = std::make_unique<Dispatcher>(ctx.store.get());

      // The context must be reachable through device() before Operations is
      // constructed: its constructor calls runtime.store(deviceId).
      devices_.push_back(std::move(ctx));
      devices_.back().operations =
          std::make_unique<Operations>(*this, devices_.size() - 1);
    }
  } catch (...) {
    devices_.clear();
    throw;
  }

  // Try to load tuning data for variant selection (non-fatal if absent)
  VariantSelector::instance().loadTuningData();
}

void Runtime::shutdown() {
  // Flush any pending commands before shutdown
  flushAllPendingCommands();
  // Destroy all device contexts (each tears down operations, dispatcher and
  // store before its interface — see DeviceContext member order).
  devices_.clear();
  // Then destroy the backend instances if present
  vulkanInstance_.reset();
  cudaInstance_.reset();
  // Reset state flags
  vulkanAvailable_ = false;
  vulkanChecked_ = false;
  cudaAvailable_ = false;
  cudaChecked_ = false;
}

BackendType Runtime::currentBackend(size_t deviceId) const {
  if (deviceId >= devices_.size()) {
    return BackendType::Vulkan; // default before initialization
  }
  return devices_[deviceId].backend;
}

Runtime::DeviceContext &Runtime::device(size_t deviceId) {
  if (deviceId >= devices_.size()) {
    throw std::runtime_error(
        "Runtime not initialized or invalid device id. Call init() first.");
  }
  return devices_[deviceId];
}

const Runtime::DeviceContext &Runtime::device(size_t deviceId) const {
  if (deviceId >= devices_.size()) {
    throw std::runtime_error(
        "Runtime not initialized or invalid device id. Call init() first.");
  }
  return devices_[deviceId];
}

ComputeInterface *Runtime::getInterface(size_t deviceId) {
  return device(deviceId).interface.get();
}

size_t Runtime::bufferCount(size_t deviceId) const {
  if (deviceId >= devices_.size()) {
    return 0;
  }
  return devices_[deviceId].interface->bufferCount();
}

size_t Runtime::activeBufferMemoryBytes(size_t deviceId) const {
  if (deviceId >= devices_.size()) {
    return 0;
  }
  return devices_[deviceId].interface->activeBufferMemoryBytes();
}

size_t Runtime::deviceTotalMemoryBytes(size_t deviceId) const {
  if (deviceId >= devices_.size()) {
    return 0;
  }
  return devices_[deviceId].interface->deviceTotalMemoryBytes();
}

void Runtime::releaseLoadingResources() {
  for (auto &ctx : devices_) {
    ctx.interface->releaseLoadingResources();
  }
}

void Runtime::flush(size_t deviceId) {
  if (devices_.empty()) {
    return;
  }
  device(deviceId).operations->flush();
  flushPendingCommands(deviceId);
}

Operations &Runtime::ops(size_t deviceId) {
  return *device(deviceId).operations;
}

TensorStore &Runtime::store(size_t deviceId) {
  return *device(deviceId).store;
}

// =========================================================================
// Tensor Operations
// =========================================================================

const ComputeBuffer &Runtime::getTensor(const Tensor &handle,
                                        size_t deviceId) const {
  return device(deviceId).store->getTensor(handle);
}

Tensor Runtime::createTensor(const std::vector<uint32_t> &shape,
                             DataType dtype,
                             const void *srcPtr,
                             bool isUniform,
                             size_t deviceId) {
  return device(deviceId).store->createTensor(shape, dtype, srcPtr, isUniform);
}

Tensor Runtime::createTensorEmpty(const std::vector<uint32_t> &shape,
                                  DataType dtype,
                                  bool isUniform,
                                  size_t deviceId) {
  return device(deviceId).store->createTensorEmpty(shape, dtype, isUniform);
}

Tensor Runtime::createTensorMapped(const std::vector<uint32_t> &shape,
                                   DataType dtype,
                                   const void *srcPtr,
                                   size_t deviceId,
                                   bool preferHost) {
  return device(deviceId).store->createTensorMapped(shape, dtype, srcPtr,
                                                    preferHost);
}

void Runtime::copyToTensor(Tensor handle,
                           const void *srcPtr,
                           size_t size,
                           size_t srcOffset,
                           size_t dstOffset,
                           size_t deviceId) {
  device(deviceId).interface->copyDataToBuffer(srcPtr, handle, size, srcOffset,
                                               dstOffset, false, true);
}

void Runtime::copyFromTensor(Tensor handle,
                             void *dstPtr,
                             size_t size,
                             size_t srcOffset,
                             size_t dstOffset,
                             size_t deviceId) {
  auto &ctx = device(deviceId);
  // Flush any pending graph operations — the executor writes results
  // directly into the placeholder buffers, so the handle is already valid.
  ctx.operations->flush();
  // Ensure all pending GPU work is complete before reading data back
  flushPendingCommands(deviceId);
  ctx.interface->copyDataFromBuffer(handle, dstPtr, size, srcOffset, dstOffset,
                                    false, true);
}

void Runtime::transferTensor(const Tensor &src,
                             size_t srcDevice,
                             const Tensor &dst,
                             size_t dstDevice) {
  const ComputeBuffer &srcBuf = device(srcDevice).store->getTensor(src);
  const ComputeBuffer &dstBuf = device(dstDevice).store->getTensor(dst);

  if (srcBuf.getDtype() != dstBuf.getDtype() ||
      srcBuf.calculateActualSize() != dstBuf.calculateActualSize()) {
    throw std::runtime_error(
        "transferTensor: source and destination dtype/size mismatch");
  }

  const size_t bytes = srcBuf.calculateActualSize();
  std::vector<uint8_t> bounce(bytes);
  copyFromTensor(src, bounce.data(), bytes, 0, 0, srcDevice);
  copyToTensor(dst, bounce.data(), bytes, 0, 0, dstDevice);
}

Tensor Runtime::transferTensor(const Tensor &src,
                               size_t srcDevice,
                               size_t dstDevice) {
  const ComputeBuffer &srcBuf = device(srcDevice).store->getTensor(src);
  Tensor dst =
      createTensorEmpty(srcBuf.getShape(), srcBuf.getDtype(), false, dstDevice);
  transferTensor(src, srcDevice, dst, dstDevice);
  return dst;
}

void Runtime::setProfilingEnabled(bool enabled) {
  profilingEnabled_ = enabled;
  for (auto &ctx : devices_) {
    ctx.interface->setProfilingEnabled(enabled);
  }
}

std::vector<DispatchTiming> Runtime::lastDispatchTimings(size_t deviceId) {
  return device(deviceId).interface->takeLastTimings();
}

// =========================================================================
// Operator Dispatch
// =========================================================================

void Runtime::dispatch(std::unique_ptr<OpNode> node, size_t deviceId) {
  dispatch(*node, deviceId);
}

void Runtime::dispatch(OpNode &node, size_t deviceId) {
  auto &ctx = device(deviceId);

  if (!ctx.dispatcher->encode(node)) {
    return;
  }

  if (isGpuBackend(ctx.backend)) {
    ctx.pendingCommands = true;
  } else {
    Tensor cmd = ctx.interface->submit();
    ctx.interface->wait(cmd);
  }
}

void Runtime::encodeBarrier(size_t deviceId) {
  if (devices_.empty()) {
    return;
  }
  auto &ctx = device(deviceId);
  ctx.dispatcher->encodeBarrier();
  ctx.pendingCommands = true;
}

void Runtime::updateBufferInline(Tensor handle,
                                 const void *data,
                                 size_t size,
                                 size_t deviceId) {
  if (devices_.empty()) {
    return;
  }
  auto &ctx = device(deviceId);
  ctx.interface->encode(ComputeDispatch::createBufferUpdate(handle, data, size));
  ctx.pendingCommands = true;
}

void Runtime::eagerSubmit(size_t deviceId) {
  if (devices_.empty()) {
    return;
  }
  auto &ctx = device(deviceId);
  if (!ctx.pendingCommands) {
    return;
  }
  // Submit the command buffer immediately so the GPU starts working.
  // The handle is stored so flushPendingCommands() can wait for it.
  ctx.pendingCmd = ctx.interface->submit();
  // pendingCommands stays true — cleared by flushPendingCommands().
}

void Runtime::flushPendingCommands(size_t deviceId) {
  if (devices_.empty()) {
    return;
  }
  auto &ctx = device(deviceId);
  if (!ctx.pendingCommands) {
    return;
  }

  if (ctx.pendingCmd) {
    // Already submitted (eager path) — just wait.
    if (profilingEnabled_) {
      auto t0 = std::chrono::high_resolution_clock::now();
      ctx.interface->wait(ctx.pendingCmd);
      auto t1 = std::chrono::high_resolution_clock::now();
      auto gpuUs =
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
              .count();
      logMsg("[Runtime] GPU wait: %lld us", gpuUs);
    } else {
      ctx.interface->wait(ctx.pendingCmd);
    }
    ctx.pendingCmd.reset();
  } else {
    // Not yet submitted — do full submit + wait.
    if (profilingEnabled_) {
      auto t0 = std::chrono::high_resolution_clock::now();
      Tensor cmd = ctx.interface->submit();
      auto t1 = std::chrono::high_resolution_clock::now();
      ctx.interface->wait(cmd);
      auto t2 = std::chrono::high_resolution_clock::now();
      auto cpuUs =
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
              .count();
      auto gpuUs =
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
              .count();
      logMsg("[Runtime] CPU submit: %lld us, GPU wait: %lld us, total: %lld us",
             cpuUs, gpuUs, cpuUs + gpuUs);
    } else {
      Tensor cmd = ctx.interface->submit();
      ctx.interface->wait(cmd);
    }
  }
  ctx.pendingCommands = false;
}

void Runtime::flushAllPendingCommands() {
  for (size_t i = 0; i < devices_.size(); ++i) {
    flushPendingCommands(i);
  }
}

ComputeHandle Runtime::submitReusable(size_t deviceId) {
  if (devices_.empty()) {
    return {};
  }
  auto &ctx = device(deviceId);
  if (!ctx.pendingCommands) {
    return {};
  }
  ctx.operations->flush();
  ComputeHandle cmd = ctx.interface->submitReusable();
  ctx.interface->wait(cmd);
  ctx.pendingCommands = false;
  return cmd;
}

void Runtime::resubmitAndWait(const ComputeHandle &cb, size_t deviceId) {
  if (devices_.empty() || !cb) {
    return;
  }
  auto &ctx = device(deviceId);
  ctx.interface->resubmit(cb);
  ctx.interface->wait(cb);
}

} // namespace cut
