#include "Runtime.h"

#include "Dispatcher.h"
#include "OpNode.h"
#include "Operations.h"
#include "VulkanCompute.h"

#include <ComputeCommon.h>
#include <chrono>
#include <stdexcept>

namespace cut {

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

void Runtime::init(BackendType backend) {
  if (backend == BackendType::Vulkan) {
    if (!isVulkanAvailable()) {
      throw std::runtime_error("Vulkan backend is not available");
    }
    interface_ = vulkanInstance_->createInterface();
    backendType_ = BackendType::Vulkan;
  } else {
    throw std::runtime_error("Invalid backend type");
  }

  // Create tensor store with the initialized interface
  store_ = std::make_unique<TensorStore>(interface_.get());

  // Create dispatcher with the tensor store
  dispatcher_ = std::make_unique<Dispatcher>(store_.get());

  // Create operations object
  operations_ = std::make_unique<Operations>(*this);
}

void Runtime::shutdown() {
  // Flush any pending commands before shutdown
  flushPendingCommands();
  // Destroy operations before dispatcher (it holds graph state and raw
  // pointers)
  operations_.reset();
  // Destroy dispatcher before interface (it holds a raw pointer)
  dispatcher_.reset();
  // Destroy tensor store before interface
  store_.reset();
  // First destroy the interface (which holds backend resources)
  interface_.reset();
  // Then destroy the Vulkan instance if present
  vulkanInstance_.reset();
  // Reset state flags
  vulkanAvailable_ = false;
  vulkanChecked_ = false;
  backendType_ = BackendType::Vulkan;
  pendingCommands_ = false;
}

size_t Runtime::bufferCount() const {
  if (!interface_) {
    return 0;
  }
  return interface_->bufferCount();
}

size_t Runtime::activeBufferMemoryBytes() const {
  if (!interface_) {
    return 0;
  }
  return interface_->activeBufferMemoryBytes();
}

void Runtime::flush() {
  if (operations_) {
    operations_->flush();
  }
  flushPendingCommands();
}

Operations &Runtime::ops() {
  if (!operations_) {
    throw std::runtime_error("Runtime not initialized. Call init() first.");
  }
  return *operations_;
}

TensorStore &Runtime::store() {
  if (!store_) {
    throw std::runtime_error("Runtime not initialized. Call init() first.");
  }
  return *store_;
}

ComputeInterface *Runtime::getInterface() {
  if (!interface_) {
    throw std::runtime_error(
        "Compute interface not initialized. Call init() first.");
  }
  return interface_.get();
}

// =========================================================================
// Tensor Operations
// =========================================================================

const ComputeBuffer &Runtime::getTensor(const Tensor &handle) const {
  return store_->getTensor(handle);
}

Tensor Runtime::createTensor(const std::vector<uint32_t> &shape,
                             DataType dtype,
                             const void *srcPtr,
                             bool isUniform) {
  return store_->createTensor(shape, dtype, srcPtr, isUniform);
}

Tensor Runtime::createTensorEmpty(const std::vector<uint32_t> &shape,
                                  DataType dtype,
                                  bool isUniform) {
  return store_->createTensorEmpty(shape, dtype, isUniform);
}

void Runtime::copyToTensor(Tensor handle,
                           const void *srcPtr,
                           size_t size,
                           size_t srcOffset,
                           size_t dstOffset) {
  getInterface()->copyDataToBuffer(srcPtr, handle, size, srcOffset, dstOffset,
                                   false, true);
}

void Runtime::copyFromTensor(Tensor handle,
                             void *dstPtr,
                             size_t size,
                             size_t srcOffset,
                             size_t dstOffset) {
  // Flush any pending graph operations — the executor writes results
  // directly into the placeholder buffers, so the handle is already valid.
  if (operations_) {
    operations_->flush();
  }
  // Ensure all pending GPU work is complete before reading data back
  flushPendingCommands();
  getInterface()->copyDataFromBuffer(handle, dstPtr, size, srcOffset, dstOffset,
                                     false, true);
}

void Runtime::setProfilingEnabled(bool enabled) {
  profilingEnabled_ = enabled;
  if (interface_) {
    interface_->setProfilingEnabled(enabled);
  }
}

// =========================================================================
// Operator Dispatch
// =========================================================================

void Runtime::dispatch(std::unique_ptr<OpNode> node) {
  dispatch(*node);
}

void Runtime::dispatch(OpNode &node) {
  if (!dispatcher_) {
    throw std::runtime_error("Dispatcher not initialized. Call init() first.");
  }

  if (!dispatcher_->encode(node)) {
    return;
  }

  if (isGpuBackend()) {
    pendingCommands_ = true;
  } else {
    Tensor cmd = getInterface()->submit();
    getInterface()->wait(cmd);
  }
}

void Runtime::eagerSubmit() {
  if (!pendingCommands_ || !interface_)
    return;
  // Submit the command buffer immediately so the GPU starts working.
  // The handle is stored so flushPendingCommands() can wait for it.
  pendingCmd_ = interface_->submit();
  // pendingCommands_ stays true — cleared by flushPendingCommands().
}

void Runtime::flushPendingCommands() {
  if (!pendingCommands_ || !interface_)
    return;

  if (pendingCmd_) {
    // Already submitted (eager path) — just wait.
    if (profilingEnabled_) {
      auto t0 = std::chrono::high_resolution_clock::now();
      interface_->wait(pendingCmd_);
      auto t1 = std::chrono::high_resolution_clock::now();
      auto gpuUs =
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
              .count();
      logMsg("[Runtime] GPU wait: %lld us", gpuUs);
    } else {
      interface_->wait(pendingCmd_);
    }
    pendingCmd_.reset();
  } else {
    // Not yet submitted — do full submit + wait.
    if (profilingEnabled_) {
      auto t0 = std::chrono::high_resolution_clock::now();
      Tensor cmd = interface_->submit();
      auto t1 = std::chrono::high_resolution_clock::now();
      interface_->wait(cmd);
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
      Tensor cmd = interface_->submit();
      interface_->wait(cmd);
    }
  }
  pendingCommands_ = false;
}

ComputeHandle Runtime::submitReusable() {
  if (!pendingCommands_ || !interface_) {
    return {};
  }
  if (operations_) {
    operations_->flush();
  }
  ComputeHandle cmd = interface_->submitReusable();
  interface_->wait(cmd);
  pendingCommands_ = false;
  return cmd;
}

void Runtime::resubmitAndWait(const ComputeHandle &cb) {
  if (!interface_ || !cb)
    return;
  interface_->resubmit(cb);
  interface_->wait(cb);
}

} // namespace cut
