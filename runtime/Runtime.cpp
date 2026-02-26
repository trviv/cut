#include "Runtime.h"

#include "Dispatcher.h"
#include "OpNode.h"
#include "Operations.h"
#include "VulkanCompute.h"

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

void Runtime::flush() {
  if (operations_ && operations_->isGraphMode()) {
    executeGraph();
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
  // If graph is still recording, execute it first
  if (operations_ && operations_->isGraphMode()) {
    executeGraph();
  }
  // If this handle was a placeholder, read from the resolved real tensor
  if (operations_) {
    for (const auto &p : operations_->resolvedTensors()) {
      if (p.first == handle) {
        handle = p.second;
        break;
      }
    }
  }
  // Ensure all pending GPU work is complete before reading data back
  flushPendingCommands();
  getInterface()->copyDataFromBuffer(handle, dstPtr, size, srcOffset, dstOffset,
                                     false, true);
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

// =========================================================================
// Graph Mode (forwarded to Operations)
// =========================================================================

void Runtime::beginGraph() {
  ops().beginGraph();
}

void Runtime::executeGraph() {
  ops().executeGraph();
}

bool Runtime::isGraphMode() const {
  return operations_ && operations_->isGraphMode();
}

void Runtime::flushPendingCommands() {
  if (pendingCommands_ && interface_) {
    Tensor cmd = interface_->submit();
    interface_->wait(cmd);
    pendingCommands_ = false;
  }
}

} // namespace cut
